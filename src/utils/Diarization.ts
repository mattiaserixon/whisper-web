import type { SpeakerAttribution, TranscriptSegment } from "./TranscriptTypes";

export interface DiarizationWindow {
    windowIndex: number;
    winStart: number;
    winEnd: number;
    ownStart: number;
    ownEnd: number;
}

export interface PyannoteRawSegment {
    id: number;
    start: number;
    end: number;
    confidence: number;
}

export interface RawWindowInterval {
    windowIndex: number;
    localId: number;
    start: number;
    end: number;
    confidence: number;
}

export interface LocalSpeakerTrack {
    key: string;
    windowIndex: number;
    localId: number;
    duration: number;
    earliestStart: number;
    embedding: number[];
}

export interface SpeakerTimelineInterval {
    start: number;
    end: number;
    kind: "speaker" | "overlap" | "unknown";
    speakerIndex?: number;
    speakerIndices?: number[];
}

/**
 * Computes the exact window start offset (in seconds) used by Transformers.js
 * AutomaticSpeechRecognitionPipeline for sliding-window Whisper inference.
 */
export function computeWhisperWindowOffset(
    windowIndex: number,
    chunkLengthSeconds: number,
    strideLengthSeconds: number,
): number {
    if (windowIndex <= 0) return 0;
    const jump = Math.max(1, chunkLengthSeconds - 2 * strideLengthSeconds);
    return windowIndex * jump;
}

/**
 * Splits a full recording duration into overlapping 10-second windows for
 * pyannote-segmentation-3.0, assigning non-overlapping interior ownership ranges.
 */
export function computeDiarizationWindows(
    totalDurationSeconds: number,
    windowSeconds = 10,
    stepSeconds = 8,
): DiarizationWindow[] {
    if (totalDurationSeconds <= 0) return [];
    if (totalDurationSeconds <= windowSeconds) {
        return [
            {
                windowIndex: 0,
                winStart: 0,
                winEnd: totalDurationSeconds,
                ownStart: 0,
                ownEnd: totalDurationSeconds,
            },
        ];
    }

    const windows: DiarizationWindow[] = [];
    let idx = 0;
    for (let start = 0; start < totalDurationSeconds; start += stepSeconds) {
        const end = Math.min(totalDurationSeconds, start + windowSeconds);
        const actualStart = Math.max(0, end - windowSeconds);
        windows.push({
            windowIndex: idx++,
            winStart: actualStart,
            winEnd: end,
            ownStart: 0,
            ownEnd: totalDurationSeconds,
        });
        if (end >= totalDurationSeconds) break;
    }

    for (let i = 0; i < windows.length; i++) {
        windows[i].ownStart =
            i === 0 ? 0 : (windows[i - 1].winEnd + windows[i].winStart) / 2;
        windows[i].ownEnd =
            i === windows.length - 1
                ? totalDurationSeconds
                : (windows[i].winEnd + windows[i + 1].winStart) / 2;
    }

    return windows;
}

export function l2Normalize(vec: ArrayLike<number>): number[] {
    let sumSq = 0;
    for (let i = 0; i < vec.length; i++) {
        sumSq += vec[i] * vec[i];
    }
    const norm = Math.sqrt(sumSq) || 1;
    const out = new Array<number>(vec.length);
    for (let i = 0; i < vec.length; i++) {
        out[i] = vec[i] / norm;
    }
    return out;
}

export function cosineSimilarity(a: number[], b: number[]): number {
    const len = Math.min(a.length, b.length);
    let dot = 0;
    let normA = 0;
    let normB = 0;
    for (let i = 0; i < len; i++) {
        dot += a[i] * b[i];
        normA += a[i] * a[i];
        normB += b[i] * b[i];
    }
    const denom = Math.sqrt(normA) * Math.sqrt(normB);
    return denom > 0 ? dot / denom : 0;
}

/**
 * Groups local speaker voice embeddings across 10s windows into consistent
 * whole-recording speakers ordered 1..N by first appearance in the audio.
 */
export function clusterSpeakerTracks(
    tracks: LocalSpeakerTrack[],
    options: {
        minReliableDuration?: number;
        mergeThreshold?: number;
        shortTrackAssignThreshold?: number;
        sameWindowMergeThreshold?: number;
    } = {},
): Map<string, number> {
    const {
        minReliableDuration = 0.45,
        mergeThreshold = 0.42,
        shortTrackAssignThreshold = 0.48,
        sameWindowMergeThreshold = 0.72,
    } = options;

    const trackToSpeaker = new Map<string, number>();
    if (tracks.length === 0) return trackToSpeaker;

    const normalizedTracks = tracks.map((t) => ({
        ...t,
        embedding: l2Normalize(t.embedding),
    }));

    const reliable = normalizedTracks.filter(
        (t) => t.duration >= minReliableDuration,
    );
    const primary = reliable.length > 0 ? reliable : normalizedTracks;
    const secondary =
        reliable.length > 0
            ? normalizedTracks.filter((t) => t.duration < minReliableDuration)
            : [];

    interface Cluster {
        members: LocalSpeakerTrack[];
        centroid: number[];
        weight: number;
        earliestStart: number;
    }

    const clusters: Cluster[] = primary.map((t) => ({
        members: [t],
        centroid: t.embedding.slice(),
        weight: t.duration,
        earliestStart: t.earliestStart,
    }));

    const sharesSubstantialSameWindow = (a: Cluster, b: Cluster): boolean => {
        for (const ma of a.members) {
            if (ma.duration < 0.9) continue;
            for (const mb of b.members) {
                if (mb.duration < 0.9) continue;
                if (
                    ma.windowIndex === mb.windowIndex &&
                    ma.localId !== mb.localId
                ) {
                    return true;
                }
            }
        }
        return false;
    };

    while (clusters.length > 1) {
        let bestI = -1;
        let bestJ = -1;
        let bestSim = -Infinity;

        for (let i = 0; i < clusters.length; i++) {
            for (let j = i + 1; j < clusters.length; j++) {
                const sim = cosineSimilarity(
                    clusters[i].centroid,
                    clusters[j].centroid,
                );
                const requiredThreshold = sharesSubstantialSameWindow(
                    clusters[i],
                    clusters[j],
                )
                    ? sameWindowMergeThreshold
                    : mergeThreshold;
                if (sim >= requiredThreshold && sim > bestSim) {
                    bestSim = sim;
                    bestI = i;
                    bestJ = j;
                }
            }
        }

        if (bestI === -1 || bestJ === -1) break;

        const target = clusters[bestI];
        const source = clusters[bestJ];
        const totalWeight = target.weight + source.weight;
        const merged = new Array<number>(target.centroid.length);
        for (let k = 0; k < merged.length; k++) {
            merged[k] =
                (target.centroid[k] * target.weight +
                    source.centroid[k] * source.weight) /
                totalWeight;
        }
        target.centroid = l2Normalize(merged);
        target.weight = totalWeight;
        target.earliestStart = Math.min(
            target.earliestStart,
            source.earliestStart,
        );
        target.members.push(...source.members);
        clusters.splice(bestJ, 1);
    }

    clusters.sort((a, b) => a.earliestStart - b.earliestStart);

    clusters.forEach((cluster, idx) => {
        const speakerIndex = idx + 1;
        for (const member of cluster.members) {
            trackToSpeaker.set(member.key, speakerIndex);
        }
    });

    for (const st of secondary) {
        let bestSpeaker: number | null = null;
        let bestSim = shortTrackAssignThreshold;
        clusters.forEach((cluster, idx) => {
            const sim = cosineSimilarity(st.embedding, cluster.centroid);
            if (sim >= bestSim) {
                bestSim = sim;
                bestSpeaker = idx + 1;
            }
        });
        if (bestSpeaker !== null) {
            trackToSpeaker.set(st.key, bestSpeaker);
        }
    }

    return trackToSpeaker;
}

const POWERSET_OVERLAP_PAIR: Record<number, [number, number]> = {
    4: [1, 2],
    5: [1, 3],
    6: [2, 3],
};

/**
 * Converts window-level pyannote powerset segments and speaker cluster assignments
 * into a global speaker timeline.
 */
export function buildSpeakerTimeline(
    rawIntervals: RawWindowInterval[],
    trackToSpeaker: Map<string, number>,
): SpeakerTimelineInterval[] {
    const timeline: SpeakerTimelineInterval[] = [];

    for (const item of rawIntervals) {
        if (item.localId === 0 || item.end <= item.start) continue;

        if (item.localId >= 1 && item.localId <= 3) {
            const speaker = trackToSpeaker.get(
                `${item.windowIndex}:${item.localId}`,
            );
            if (speaker !== undefined) {
                timeline.push({
                    start: item.start,
                    end: item.end,
                    kind: "speaker",
                    speakerIndex: speaker,
                });
            } else {
                timeline.push({
                    start: item.start,
                    end: item.end,
                    kind: "unknown",
                });
            }
        } else if (item.localId in POWERSET_OVERLAP_PAIR) {
            const [l1, l2] = POWERSET_OVERLAP_PAIR[item.localId];
            const s1 = trackToSpeaker.get(`${item.windowIndex}:${l1}`);
            const s2 = trackToSpeaker.get(`${item.windowIndex}:${l2}`);
            const uniqueSpeakers = Array.from(
                new Set([s1, s2].filter((v): v is number => v !== undefined)),
            ).sort((a, b) => a - b);

            if (uniqueSpeakers.length === 1 && s1 !== undefined && s2 !== undefined) {
                // Both local tracks resolved to the same speaker cluster
                timeline.push({
                    start: item.start,
                    end: item.end,
                    kind: "speaker",
                    speakerIndex: uniqueSpeakers[0],
                });
            } else {
                timeline.push({
                    start: item.start,
                    end: item.end,
                    kind: "overlap",
                    speakerIndices:
                        uniqueSpeakers.length > 0 ? uniqueSpeakers : undefined,
                });
            }
        }
    }

    timeline.sort((a, b) => a.start - b.start);
    return timeline;
}

function attributeRange(
    segStart: number,
    segEnd: number,
    timeline: SpeakerTimelineInterval[],
): { attribution: SpeakerAttribution; totalSpeech: number } {
    const speakerDurations = new Map<number, number>();
    const speakerFirstSeen = new Map<number, number>();
    const overlapSpeakers = new Set<number>();
    let overlapDuration = 0;
    let unknownDuration = 0;

    for (const iv of timeline) {
        if (iv.end <= segStart) continue;
        if (iv.start >= segEnd) break;
        const ovStart = Math.max(segStart, iv.start);
        const ovEnd = Math.min(segEnd, iv.end);
        const dur = ovEnd - ovStart;
        if (dur <= 0) continue;

        if (iv.kind === "speaker" && iv.speakerIndex !== undefined) {
            speakerDurations.set(
                iv.speakerIndex,
                (speakerDurations.get(iv.speakerIndex) ?? 0) + dur,
            );
            if (!speakerFirstSeen.has(iv.speakerIndex)) {
                speakerFirstSeen.set(iv.speakerIndex, ovStart);
            }
        } else if (iv.kind === "overlap") {
            overlapDuration += dur;
            iv.speakerIndices?.forEach((s) => overlapSpeakers.add(s));
        } else {
            unknownDuration += dur;
        }
    }

    let knownSpeakerSum = 0;
    for (const d of speakerDurations.values()) knownSpeakerSum += d;
    const totalSpeech = knownSpeakerSum + overlapDuration + unknownDuration;

    if (totalSpeech <= 0.04) {
        return { attribution: { kind: "unknown" }, totalSpeech };
    }

    if (overlapDuration / totalSpeech >= 0.35) {
        const indices = Array.from(
            new Set([...overlapSpeakers, ...speakerDurations.keys()]),
        ).sort((a, b) => a - b);
        return {
            attribution: {
                kind: "overlap",
                speakerIndices: indices.length > 0 ? indices : undefined,
            },
            totalSpeech,
        };
    }

    const rankedSpeakers = Array.from(speakerDurations.entries()).sort(
        (a, b) => b[1] - a[1],
    );

    if (rankedSpeakers.length === 0) {
        return { attribution: { kind: "unknown" }, totalSpeech };
    }

    const [topSpeaker, topDur] = rankedSpeakers[0];
    const secondDur = rankedSpeakers.length > 1 ? rankedSpeakers[1][1] : 0;

    if (topDur / totalSpeech >= 0.65 && secondDur / totalSpeech < 0.3) {
        return {
            attribution: { kind: "speaker", speakerIndex: topSpeaker },
            totalSpeech,
        };
    }

    if (
        rankedSpeakers.length >= 2 &&
        (topDur + secondDur) / totalSpeech >= 0.65 &&
        (secondDur >= 0.35 || secondDur / totalSpeech >= 0.25)
    ) {
        const majorSpeakers = rankedSpeakers
            .filter(([, d]) => d / totalSpeech >= 0.2 || d >= 0.35)
            .map(([spk]) => spk)
            .sort(
                (a, b) =>
                    (speakerFirstSeen.get(a) ?? 0) -
                    (speakerFirstSeen.get(b) ?? 0),
            );
        return {
            attribution: {
                kind: "mixed",
                speakerIndices: majorSpeakers,
            },
            totalSpeech,
        };
    }

    return { attribution: { kind: "unknown" }, totalSpeech };
}

/**
 * Enriches Whisper transcript segments with speaker attributions without
 * modifying or guessing word splits inside Whisper's text.
 */
export function alignSegmentsWithSpeakers(
    segments: TranscriptSegment[],
    timeline: SpeakerTimelineInterval[],
    audioDurationSeconds: number,
): TranscriptSegment[] {
    return segments.map((seg, idx) => {
        if (seg.startSeconds === null) {
            return { ...seg, speaker: { kind: "unknown" } };
        }

        const nextStart =
            idx + 1 < segments.length ? segments[idx + 1].startSeconds : null;
        const fallbackEnd =
            nextStart !== null && nextStart > seg.startSeconds
                ? nextStart
                : Math.max(seg.startSeconds + 0.5, audioDurationSeconds);
        const segEnd =
            seg.endSeconds !== null && seg.endSeconds > seg.startSeconds
                ? seg.endSeconds
                : fallbackEnd;

        const initialResult = attributeRange(
            seg.startSeconds,
            segEnd,
            timeline,
        );
        let attribution = initialResult.attribution;

        // If Whisper's timestamp slightly missed a very short utterance, check a small ±0.3s collar
        if (initialResult.totalSpeech <= 0.04) {
            const collarStart = Math.max(0, seg.startSeconds - 0.3);
            const collarEnd = Math.min(audioDurationSeconds, segEnd + 0.3);
            const retry = attributeRange(collarStart, collarEnd, timeline);
            if (retry.totalSpeech > 0.04) {
                attribution = retry.attribution;
            }
        }

        return {
            ...seg,
            speaker: attribution,
        };
    });
}
