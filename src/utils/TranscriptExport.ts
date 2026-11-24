import {
    formatDisplayTimeRange,
    formatSrtTimeRange,
    resolveSegmentEndSeconds,
} from "./AudioUtils";
import { SpeakerAttribution, TranscriptSegment } from "./TranscriptTypes";

export interface SpeakerLabelTranslations {
    speakerN: (n: number) => string;
    overlappingSpeech: string;
    multipleSpeakers: string;
    unknownSpeaker: string;
}

/**
 * Formats a localized human-readable speaker label from structured speaker metadata.
 */
export function formatSpeakerAttribution(
    speaker: SpeakerAttribution | null | undefined,
    labels: SpeakerLabelTranslations,
): string | null {
    if (!speaker) return null;

    if (speaker.kind === "speaker" && speaker.speakerIndex !== undefined) {
        return labels.speakerN(speaker.speakerIndex);
    }

    if (speaker.kind === "overlap") {
        if (speaker.speakerIndices && speaker.speakerIndices.length >= 2) {
            return speaker.speakerIndices
                .map((n: number) => labels.speakerN(n))
                .join(" + ");
        }
        return labels.overlappingSpeech;
    }

    if (speaker.kind === "mixed") {
        if (speaker.speakerIndices && speaker.speakerIndices.length >= 2) {
            return speaker.speakerIndices
                .map((n: number) => labels.speakerN(n))
                .join(" / ");
        }
        return labels.multipleSpeakers;
    }

    return labels.unknownSpeaker;
}

function getSpeakerId(speaker: SpeakerAttribution): string {
    if (speaker.kind === "speaker" && speaker.speakerIndex !== undefined) {
        return `SPEAKER_${speaker.speakerIndex}`;
    }
    if (speaker.kind === "overlap") {
        return "OVERLAP";
    }
    if (speaker.kind === "mixed") {
        return "MIXED";
    }
    return "UNKNOWN";
}

/**
 * Plain continuous text export (preserves backwards-compatible plain TXT behavior).
 */
export function buildPlainTxtExport(segments: TranscriptSegment[]): string {
    return segments
        .map((seg) => seg.text)
        .join("")
        .trim();
}

/**
 * Structured, human-readable timestamped (and optionally speaker-labeled) text export.
 */
export function buildStructuredTxtExport(
    segments: TranscriptSegment[],
    labels: SpeakerLabelTranslations,
): string {
    const includeHours = segments.some(
        (s) =>
            (s.startSeconds !== null && s.startSeconds >= 3600) ||
            (s.endSeconds !== null && s.endSeconds >= 3600),
    );

    return segments
        .map((seg) => {
            const cleanText = seg.text.trim();
            if (!cleanText) return "";

            const range =
                formatDisplayTimeRange(
                    seg.startSeconds,
                    seg.endSeconds,
                    includeHours,
                ) ?? "—";
            const speakerLabel = formatSpeakerAttribution(seg.speaker, labels);

            if (speakerLabel) {
                return `[${range}] ${speakerLabel}: ${cleanText}`;
            }
            return `[${range}] ${cleanText}`;
        })
        .filter((line) => line.length > 0)
        .join("\n");
}

/**
 * Structured JSON export preserving backwards-compatible `text` and `timestamp` fields
 * while adding `id`, `startSeconds`, `endSeconds`, and `speaker`.
 */
export function buildJsonExport(
    segments: TranscriptSegment[],
    labels: SpeakerLabelTranslations,
): string {
    const payload = segments.map((seg, idx) => {
        const speakerLabel = formatSpeakerAttribution(seg.speaker, labels);
        return {
            id: seg.id || `seg-${idx}`,
            text: seg.text,
            timestamp: seg.timestamp,
            startSeconds: seg.startSeconds,
            endSeconds: seg.endSeconds,
            speaker: seg.speaker
                ? {
                      id: getSpeakerId(seg.speaker),
                      label: speakerLabel,
                      kind: seg.speaker.kind,
                      ...(seg.speaker.speakerIndex !== undefined
                          ? { speakerIndex: seg.speaker.speakerIndex }
                          : {}),
                      ...(seg.speaker.speakerIndices
                          ? { speakerIndices: seg.speaker.speakerIndices }
                          : {}),
                  }
                : null,
        };
    });

    let jsonData = JSON.stringify(payload, null, 2);
    // Keep compact single-line [start, end] arrays for readability
    const regex = /( {4}"timestamp": )\[\s+(\S+)\s+(\S+)\s+\]/gm;
    jsonData = jsonData.replace(regex, "$1[$2 $3]");
    return jsonData;
}

/**
 * Valid SRT subtitle export with monotonic cue numbering, strictly positive durations
 * (`end > start`), and optional speaker prefixes.
 */
export function buildSrtExport(
    segments: TranscriptSegment[],
    labels: SpeakerLabelTranslations,
    audioDuration?: number,
): string {
    const cues: string[] = [];
    let cueNumber = 1;

    for (let i = 0; i < segments.length; i++) {
        const seg = segments[i];
        const cleanText = seg.text.trim();
        if (!cleanText || seg.startSeconds === null) {
            continue;
        }

        let nextStart: number | null = null;
        for (let j = i + 1; j < segments.length; j++) {
            if (segments[j].startSeconds !== null) {
                nextStart = segments[j].startSeconds;
                break;
            }
        }

        const { endSeconds: resolvedEnd } = resolveSegmentEndSeconds(
            seg.startSeconds,
            seg.endSeconds,
            nextStart,
            audioDuration ?? null,
            cleanText,
        );

        const speakerLabel = formatSpeakerAttribution(seg.speaker, labels);
        const body = speakerLabel ? `[${speakerLabel}] ${cleanText}` : cleanText;

        cues.push(
            `${cueNumber}\n${formatSrtTimeRange(seg.startSeconds, resolvedEnd)}\n${body}\n`,
        );
        cueNumber += 1;
    }

    return cues.join("\n");
}
