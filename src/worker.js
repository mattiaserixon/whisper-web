import {
    pipeline,
    WhisperTextStreamer,
    AutoProcessor,
    AutoModelForAudioFrameClassification,
    AutoModel,
} from "@huggingface/transformers";
import Constants from "./utils/Constants";
import {
    computeWhisperWindowOffset,
    computeDiarizationWindows,
    clusterSpeakerTracks,
    buildSpeakerTimeline,
    alignSegmentsWithSpeakers,
    l2Normalize,
} from "./utils/Diarization";

const SAMPLING_RATE = Constants.SAMPLING_RATE;

// Track active and cancelled request IDs so user cancellations and input changes take effect immediately
let latestRequestId = 0;
const cancelledRequestIds = new Set();

function isRequestCancelled(requestId) {
    return (
        cancelledRequestIds.has(requestId) ||
        (requestId !== undefined && requestId !== latestRequestId)
    );
}

// Define model factories
// Ensures only one model is created of each type
class PipelineFactory {
    static task = null;
    static model = null;
    static dtype = null;
    static gpu = false;
    static instance = null;

    constructor(tokenizer, model, dtype, gpu) {
        this.tokenizer = tokenizer;
        this.model = model;
        this.dtype = dtype;
        this.gpu = gpu;
    }

    static async getInstance(progress_callback = null) {
        if (this.instance === null) {
            this.instance = pipeline(this.task, this.model, {
                dtype: this.dtype,
                device: this.gpu ? "webgpu" : "wasm",
                progress_callback,
            });
        }

        return this.instance;
    }
}

class AutomaticSpeechRecognitionPipelineFactory extends PipelineFactory {
    static task = "automatic-speech-recognition";
    static model = null;
    static dtype = null;
    static gpu = false;
}

class SpeakerDiarizationFactory {
    static gpu = false;
    static segProcessor = null;
    static segModel = null;
    static embProcessor = null;
    static embModel = null;
    static loadingPromise = null;

    static async loadModels(device, progress_callback) {
        const segId = Constants.DIARIZATION_SEGMENTATION_MODEL;
        const embId = Constants.DIARIZATION_EMBEDDING_MODEL;

        const [segProcessor, segModel, embProcessor, embModel] =
            await Promise.all([
                AutoProcessor.from_pretrained(segId, { progress_callback }),
                AutoModelForAudioFrameClassification.from_pretrained(segId, {
                    dtype: "fp32",
                    device,
                    progress_callback,
                }),
                AutoProcessor.from_pretrained(embId, { progress_callback }),
                AutoModel.from_pretrained(embId, {
                    dtype: "q8",
                    device,
                    progress_callback,
                }),
            ]);

        return { segProcessor, segModel, embProcessor, embModel };
    }

    static async getInstances(gpu = false, progress_callback = null) {
        if (
            this.segModel !== null &&
            this.embModel !== null &&
            this.gpu === gpu
        ) {
            return {
                segProcessor: this.segProcessor,
                segModel: this.segModel,
                embProcessor: this.embProcessor,
                embModel: this.embModel,
            };
        }

        if (this.loadingPromise !== null && this.gpu === gpu) {
            return this.loadingPromise;
        }

        // Dispose previous instances if device setting changed
        if (this.segModel !== null) {
            await this.segModel.dispose?.();
            this.segModel = null;
        }
        if (this.embModel !== null) {
            await this.embModel.dispose?.();
            this.embModel = null;
        }

        this.gpu = gpu;
        const preferredDevice = gpu ? "webgpu" : "wasm";

        this.loadingPromise = (async () => {
            try {
                const loaded = await this.loadModels(
                    preferredDevice,
                    progress_callback,
                );
                Object.assign(this, loaded);
                return loaded;
            } catch (err) {
                if (preferredDevice === "webgpu") {
                    // Fallback to WASM if WebGPU lacks an operator needed by speaker models
                    const loaded = await this.loadModels(
                        "wasm",
                        progress_callback,
                    );
                    this.gpu = false;
                    Object.assign(this, loaded);
                    return loaded;
                }
                throw err;
            } finally {
                this.loadingPromise = null;
            }
        })();

        return this.loadingPromise;
    }
}

function normalizeFinalSegments(rawChunks, fallbackText) {
    if (!Array.isArray(rawChunks) || rawChunks.length === 0) {
        const trimmed = (fallbackText ?? "").trim();
        if (!trimmed) return [];
        return [
            {
                id: "seg-0",
                text: trimmed,
                startSeconds: null,
                endSeconds: null,
                timestamp: [null, null],
                speaker: null,
                finalised: true,
            },
        ];
    }

    const normalized = [];
    for (let i = 0; i < rawChunks.length; i++) {
        const raw = rawChunks[i];
        const text = typeof raw?.text === "string" ? raw.text : "";
        if (text.trim().length === 0) continue;

        const rawStart = raw?.timestamp?.[0];
        const rawEnd = raw?.timestamp?.[1];
        const startSeconds =
            typeof rawStart === "number" && !Number.isNaN(rawStart)
                ? rawStart
                : null;
        const endSeconds =
            typeof rawEnd === "number" && !Number.isNaN(rawEnd) ? rawEnd : null;

        normalized.push({
            id: `seg-${normalized.length}`,
            text,
            startSeconds,
            endSeconds,
            timestamp: [startSeconds, endSeconds],
            speaker: null,
            finalised: true,
        });
    }

    normalized.sort((a, b) => {
        if (a.startSeconds === null && b.startSeconds === null) return 0;
        if (a.startSeconds === null) return 1;
        if (b.startSeconds === null) return -1;
        return a.startSeconds - b.startSeconds;
    });

    return normalized.map((seg, idx) => ({
        ...seg,
        id: `seg-${idx}`,
    }));
}

async function runSpeakerDiarization({
    audio,
    segments,
    gpu,
    requestId,
}) {
    const rawIntervals = [];
    const tracks = [];

    try {
        self.postMessage({
            status: "diarization_status",
            requestId,
            diarizationStatus: "loading_models",
            progress: 0,
        });

        const { segProcessor, segModel, embProcessor, embModel } =
            await SpeakerDiarizationFactory.getInstances(gpu, (data) => {
                if (!isRequestCancelled(requestId)) {
                    self.postMessage({ ...data, requestId });
                }
            });

        self.postMessage({
            status: "ready",
            requestId,
        });

        if (isRequestCancelled(requestId)) {
            self.postMessage({
                status: "diarization_status",
                requestId,
                diarizationStatus: "cancelled",
            });
            return;
        }

        const totalDuration = audio.length / SAMPLING_RATE;
        if (totalDuration <= 0 || segments.length === 0) {
            self.postMessage({
                status: "diarization_complete",
                requestId,
                data: { chunks: segments },
            });
            return;
        }

        const windows = computeDiarizationWindows(totalDuration, 10, 8);

        self.postMessage({
            status: "diarization_status",
            requestId,
            diarizationStatus: "analyzing",
            progress: 0,
        });

        for (let w = 0; w < windows.length; w++) {
            if (isRequestCancelled(requestId)) {
                self.postMessage({
                    status: "diarization_status",
                    requestId,
                    diarizationStatus: "cancelled",
                });
                return;
            }

            const { windowIndex, winStart, winEnd, ownStart, ownEnd } =
                windows[w];
            const startSample = Math.max(0, Math.floor(winStart * SAMPLING_RATE));
            const endSample = Math.min(
                audio.length,
                Math.ceil(winEnd * SAMPLING_RATE),
            );
            const windowAudio = audio.subarray(startSample, endSample);
            if (windowAudio.length < 400) continue;

            const segInputs = await segProcessor(windowAudio);
            const { logits } = await segModel(segInputs);
            const diarized = segProcessor.post_process_speaker_diarization(
                logits,
                windowAudio.length,
            )[0];

            if (isRequestCancelled(requestId)) {
                self.postMessage({
                    status: "diarization_status",
                    requestId,
                    diarizationStatus: "cancelled",
                });
                return;
            }

            const cleanSegmentsByLocalSpeaker = new Map();

            for (const seg of diarized) {
                if (seg.id === 0) continue;
                const absStart = winStart + seg.start;
                const absEnd = winStart + seg.end;

                // Collect non-overlapping single-speaker regions (ids 1..3) for clean voice embeddings
                if (seg.id >= 1 && seg.id <= 3) {
                    if (!cleanSegmentsByLocalSpeaker.has(seg.id)) {
                        cleanSegmentsByLocalSpeaker.set(seg.id, []);
                    }
                    cleanSegmentsByLocalSpeaker.get(seg.id).push({
                        start: absStart,
                        end: absEnd,
                        confidence: seg.confidence,
                    });
                }

                const clipStart = Math.max(ownStart, absStart);
                const clipEnd = Math.min(ownEnd, absEnd);
                if (clipEnd > clipStart) {
                    rawIntervals.push({
                        windowIndex,
                        localId: seg.id,
                        start: clipStart,
                        end: clipEnd,
                        confidence: seg.confidence,
                    });
                }
            }

            for (const [
                localId,
                localSegs,
            ] of cleanSegmentsByLocalSpeaker.entries()) {
                if (isRequestCancelled(requestId)) {
                    self.postMessage({
                        status: "diarization_status",
                        requestId,
                        diarizationStatus: "cancelled",
                    });
                    return;
                }

                let totalSamples = 0;
                const slices = [];
                for (const ls of localSegs) {
                    const s0 = Math.max(
                        0,
                        Math.floor(ls.start * SAMPLING_RATE),
                    );
                    const s1 = Math.min(
                        audio.length,
                        Math.ceil(ls.end * SAMPLING_RATE),
                    );
                    if (s1 > s0) {
                        slices.push(audio.subarray(s0, s1));
                        totalSamples += s1 - s0;
                    }
                }

                const cleanDuration = totalSamples / SAMPLING_RATE;
                // Require at least 200ms of clean speech to extract an embedding
                if (cleanDuration >= 0.2) {
                    // Cap embedding input at 8s of clean speech per window speaker for fast CPU execution
                    const maxSamples = Math.min(
                        totalSamples,
                        SAMPLING_RATE * 8,
                    );
                    const concat = new Float32Array(maxSamples);
                    let pos = 0;
                    for (const sl of slices) {
                        const take = Math.min(sl.length, maxSamples - pos);
                        if (take <= 0) break;
                        concat.set(sl.subarray(0, take), pos);
                        pos += take;
                    }

                    const embInputs = await embProcessor(concat);
                    const embOutput = await embModel(embInputs);
                    const embeddingTensor =
                        embOutput.last_hidden_state ?? embOutput.embeddings;

                    if (embeddingTensor?.data) {
                        tracks.push({
                            key: `${windowIndex}:${localId}`,
                            windowIndex,
                            localId,
                            duration: cleanDuration,
                            earliestStart: localSegs[0].start,
                            embedding: l2Normalize(
                                Array.from(embeddingTensor.data),
                            ),
                        });
                    }
                }
            }

            const progress = Math.round(((w + 1) / windows.length) * 100);
            self.postMessage({
                status: "diarization_status",
                requestId,
                diarizationStatus: "analyzing",
                progress,
            });
        }

        if (isRequestCancelled(requestId)) {
            self.postMessage({
                status: "diarization_status",
                requestId,
                diarizationStatus: "cancelled",
            });
            return;
        }

        const trackToSpeaker = clusterSpeakerTracks(tracks);
        const timeline = buildSpeakerTimeline(rawIntervals, trackToSpeaker);
        const enrichedSegments = alignSegmentsWithSpeakers(
            segments,
            timeline,
            totalDuration,
        );

        self.postMessage({
            status: "diarization_complete",
            requestId,
            data: {
                chunks: enrichedSegments,
            },
        });
    } catch (error) {
        console.error("Speaker diarization failed:", error);
        if (!isRequestCancelled(requestId)) {
            self.postMessage({
                status: "diarization_unavailable",
                requestId,
                error: error?.message ?? String(error),
            });
        }
    } finally {
        // Explicitly wipe temporary voice embeddings and intervals from memory
        for (const t of tracks) {
            t.embedding.fill(0);
        }
        tracks.length = 0;
        rawIntervals.length = 0;
    }
}

self.addEventListener("message", async (event) => {
    const message = event.data;
    if (!message) return;

    if (message.type === "cancel_diarization" || message.type === "cancel") {
        if (message.requestId !== undefined) {
            cancelledRequestIds.add(message.requestId);
        }
        return;
    }

    const requestId = message.requestId ?? ++latestRequestId;
    latestRequestId = requestId;

    const transcript = await transcribe({ ...message, requestId });
    if (transcript === null || isRequestCancelled(requestId)) return;

    const finalSegments = normalizeFinalSegments(
        transcript.chunks,
        transcript.text,
    );

    // Send authoritative merged Whisper segments immediately so the timed transcript is usable right away
    self.postMessage({
        status: "complete",
        requestId,
        data: {
            tps: transcript.tps,
            text: transcript.text ?? "",
            chunks: finalSegments,
            willDiarize: Boolean(message.diarize),
        },
    });

    if (message.diarize && !isRequestCancelled(requestId)) {
        await runSpeakerDiarization({
            audio: message.audio,
            segments: finalSegments,
            gpu: Boolean(message.gpu),
            requestId,
        });
    }
});

const transcribe = async ({
    audio,
    model,
    dtype,
    gpu,
    subtask,
    language,
    requestId,
}) => {
    const isDistilWhisper = model.startsWith("distil-whisper/");

    const p = AutomaticSpeechRecognitionPipelineFactory;
    if (p.model !== model || p.dtype !== dtype || p.gpu !== gpu) {
        // Invalidate model if different model, dtype, or gpu setting
        p.model = model;
        p.dtype = dtype;
        p.gpu = gpu;

        if (p.instance !== null) {
            (await p.getInstance()).dispose();
            p.instance = null;
        }
    }

    // Load transcriber model
    const transcriber = await p.getInstance((data) => {
        self.postMessage({ ...data, requestId });
    });

    const time_precision =
        transcriber.processor.feature_extractor.config.chunk_length /
        transcriber.model.config.max_source_positions;

    /** @type {import("./utils/TranscriptTypes").TranscriptSegment[]} */
    const liveChunks = [];

    const chunk_length_s = isDistilWhisper ? 20 : 30;
    const stride_length_s = isDistilWhisper ? 3 : 5;

    let chunk_count = 0;
    let start_time;
    let num_tokens = 0;
    let tps;
    const streamer = new WhisperTextStreamer(transcriber.tokenizer, {
        time_precision,
        on_chunk_start: (x) => {
            const offset = computeWhisperWindowOffset(
                chunk_count,
                chunk_length_s,
                stride_length_s,
            );
            const startSeconds = Math.round((offset + x) * 100) / 100;
            liveChunks.push({
                id: `live-${liveChunks.length}`,
                text: "",
                startSeconds,
                endSeconds: null,
                timestamp: [startSeconds, null],
                speaker: null,
                finalised: false,
            });
        },
        token_callback_function: () => {
            start_time ??= performance.now();
            if (num_tokens++ > 0) {
                tps = (num_tokens / (performance.now() - start_time)) * 1000;
            }
        },
        callback_function: (x) => {
            if (liveChunks.length === 0 || isRequestCancelled(requestId)) return;
            liveChunks.at(-1).text += x;

            self.postMessage({
                status: "update",
                requestId,
                data: {
                    text: "",
                    chunks: liveChunks,
                    tps,
                },
            });
        },
        on_chunk_end: (x) => {
            if (liveChunks.length === 0) return;
            const offset = computeWhisperWindowOffset(
                chunk_count,
                chunk_length_s,
                stride_length_s,
            );
            const current = liveChunks.at(-1);
            const endSeconds = Math.round((offset + x) * 100) / 100;
            current.endSeconds = endSeconds;
            current.timestamp[1] = endSeconds;
            current.finalised = true;
        },
        on_finalize: () => {
            start_time = null;
            num_tokens = 0;
            ++chunk_count;
        },
    });

    // Actually run transcription
    const output = await transcriber(audio, {
        // Greedy
        top_k: 0,
        do_sample: false,

        // Sliding window
        chunk_length_s,
        stride_length_s,

        // Language and task
        language,
        task: subtask,

        // Return timestamps
        return_timestamps: true,
        force_full_sequences: false,

        // Callback functions
        streamer, // after each generation step
    }).catch((error) => {
        console.error(error);
        self.postMessage({
            status: "error",
            requestId,
            data: error,
        });
        return null;
    });

    if (output === null) return null;

    return {
        tps,
        ...output,
    };
};
