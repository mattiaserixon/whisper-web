import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useWorker } from "./useWorker";
import { useTranslation } from "react-i18next";
import Constants from "../utils/Constants";
import {
    DiarizationStatus,
    TranscriptSegment,
} from "../utils/TranscriptTypes";

export interface ProgressItem {
    file: string;
    loaded: number;
    progress: number;
    total: number;
    name: string;
    status: string;
}

interface TranscriberUpdateData {
    requestId?: number;
    status: "update" | "complete";
    data: {
        text: string;
        chunks: TranscriptSegment[];
        tps?: number;
        willDiarize?: boolean;
    };
}

export interface TranscriberData {
    isBusy: boolean;
    tps?: number;
    text: string;
    chunks: TranscriptSegment[];
    audioDuration?: number;
}

export interface Transcriber {
    onInputChange: () => void;
    isBusy: boolean;
    isModelLoading: boolean;
    progressItems: ProgressItem[];
    start: (audioData: AudioBuffer | undefined) => void;
    output?: TranscriberData;
    model: string;
    setModel: (model: string) => void;
    dtype: string;
    setDtype: (dtype: string) => void;
    gpu: boolean;
    setGPU: (gpu: boolean) => void;
    subtask: string;
    setSubtask: (subtask: string) => void;
    language?: string;
    setLanguage: (language: string) => void;
    diarize: boolean;
    setDiarize: (diarize: boolean) => void;
    diarizationStatus: DiarizationStatus;
    diarizationProgress: number;
    diarizationError?: string;
    cancelDiarization: () => void;
}

function isSameProgressFile(item: ProgressItem, message: ProgressItem): boolean {
    if (item.name && message.name) {
        return item.name === message.name && item.file === message.file;
    }
    return item.file === message.file;
}

export function useTranscriber(): Transcriber {
    const [transcript, setTranscript] = useState<TranscriberData | undefined>(
        undefined,
    );
    const [isBusy, setIsBusy] = useState(false);
    const [isModelLoading, setIsModelLoading] = useState(false);
    const [progressItems, setProgressItems] = useState<ProgressItem[]>([]);

    const [diarize, setDiarize] = useState<boolean>(Constants.DEFAULT_DIARIZE);
    const [diarizationStatus, setDiarizationStatus] =
        useState<DiarizationStatus>("disabled");
    const [diarizationProgress, setDiarizationProgress] = useState<number>(0);
    const [diarizationError, setDiarizationError] = useState<
        string | undefined
    >(undefined);

    const activeRequestIdRef = useRef<number>(0);
    const activeAudioDurationRef = useRef<number | undefined>(undefined);

    const webWorker = useWorker((event) => {
        const message = event.data;
        if (!message) return;

        // Ignore stale worker events from superseded or cancelled requests
        if (
            message.requestId !== undefined &&
            message.requestId !== activeRequestIdRef.current
        ) {
            return;
        }

        switch (message.status) {
            case "progress":
                setProgressItems((prev) =>
                    prev.map((item) => {
                        if (isSameProgressFile(item, message)) {
                            return { ...item, progress: message.progress };
                        }
                        return item;
                    }),
                );
                break;
            case "update":
            case "complete": {
                const busy = message.status === "update";
                const updateMessage = message as TranscriberUpdateData;
                setTranscript({
                    isBusy: busy,
                    text: updateMessage.data.text,
                    tps: updateMessage.data.tps,
                    chunks: updateMessage.data.chunks,
                    audioDuration: activeAudioDurationRef.current,
                });
                setIsBusy(busy);
                if (message.status === "complete") {
                    if (updateMessage.data.willDiarize) {
                        setDiarizationStatus("loading_models");
                        setDiarizationProgress(0);
                    } else {
                        setDiarizationStatus("disabled");
                    }
                }
                break;
            }
            case "diarization_status": {
                const nextStatus =
                    message.diarizationStatus as DiarizationStatus;
                setDiarizationStatus(nextStatus);
                if (typeof message.progress === "number") {
                    setDiarizationProgress(message.progress);
                }
                if (nextStatus === "cancelled") {
                    setIsModelLoading(false);
                    setProgressItems([]);
                }
                break;
            }
            case "diarization_complete": {
                setTranscript((prev) =>
                    prev
                        ? {
                              ...prev,
                              chunks: message.data.chunks,
                          }
                        : prev,
                );
                setDiarizationStatus("ready");
                setDiarizationProgress(100);
                setIsModelLoading(false);
                setProgressItems([]);
                break;
            }
            case "diarization_unavailable": {
                setDiarizationStatus("unavailable");
                setDiarizationError(message.error);
                setIsModelLoading(false);
                setProgressItems([]);
                break;
            }
            case "initiate":
                setIsModelLoading(true);
                setProgressItems((prev) => {
                    if (prev.some((item) => isSameProgressFile(item, message))) {
                        return prev;
                    }
                    return [...prev, message];
                });
                break;
            case "ready":
                setIsModelLoading(false);
                break;
            case "error":
                setIsBusy(false);
                setIsModelLoading(false);
                setDiarizationStatus("disabled");
                alert(
                    `An error occurred: "${message.data.message}". Please file a bug report.`,
                );
                break;
            case "done":
                setProgressItems((prev) => {
                    const remaining = prev.filter(
                        (item) => !isSameProgressFile(item, message),
                    );
                    if (remaining.length === 0) {
                        setIsModelLoading(false);
                    }
                    return remaining;
                });
                break;

            default:
                break;
        }
    });

    const { i18n } = useTranslation();

    const [model, setModel] = useState<string>(() =>
        Constants.getDefaultModel(i18n.language),
    );

    const [subtask, setSubtask] = useState<string>(Constants.DEFAULT_SUBTASK);
    const [dtype, setDtype] = useState<string>(Constants.DEFAULT_DTYPE);
    const [gpu, setGPU] = useState<boolean>(Constants.DEFAULT_GPU);
    const [language, setLanguage] = useState<string>(() =>
        Constants.getDefaultLanguage(i18n.language),
    );

    useEffect(() => {
        const handleLanguageChanged = (lng: string) => {
            setModel(Constants.getDefaultModel(lng));
            setLanguage(Constants.getDefaultLanguage(lng));
        };

        i18n.on("languageChanged", handleLanguageChanged);
        return () => {
            i18n.off("languageChanged", handleLanguageChanged);
        };
    }, [i18n]);

    const cancelDiarization = useCallback(() => {
        if (activeRequestIdRef.current > 0) {
            webWorker.postMessage({
                type: "cancel_diarization",
                requestId: activeRequestIdRef.current,
            });
        }
        setDiarizationStatus("cancelled");
        setIsModelLoading(false);
        setProgressItems([]);
    }, [webWorker]);

    const onInputChange = useCallback(() => {
        if (activeRequestIdRef.current > 0) {
            webWorker.postMessage({
                type: "cancel",
                requestId: activeRequestIdRef.current,
            });
        }
        activeRequestIdRef.current += 1;
        activeAudioDurationRef.current = undefined;
        setTranscript(undefined);
        setIsBusy(false);
        setIsModelLoading(false);
        setProgressItems([]);
        setDiarizationStatus("disabled");
        setDiarizationProgress(0);
        setDiarizationError(undefined);
    }, [webWorker]);

    const postRequest = useCallback(
        async (audioData: AudioBuffer | undefined) => {
            if (audioData) {
                if (activeRequestIdRef.current > 0) {
                    webWorker.postMessage({
                        type: "cancel",
                        requestId: activeRequestIdRef.current,
                    });
                }

                const requestId = activeRequestIdRef.current + 1;
                activeRequestIdRef.current = requestId;
                activeAudioDurationRef.current = audioData.duration;

                setTranscript(undefined);
                setIsBusy(true);
                setDiarizationError(undefined);
                setDiarizationProgress(0);
                setDiarizationStatus(diarize ? "pending" : "disabled");

                let audio;
                if (audioData.numberOfChannels === 2) {
                    const SCALING_FACTOR = Math.sqrt(2);

                    const left = audioData.getChannelData(0);
                    const right = audioData.getChannelData(1);

                    audio = new Float32Array(left.length);
                    for (let i = 0; i < audioData.length; ++i) {
                        audio[i] = (SCALING_FACTOR * (left[i] + right[i])) / 2;
                    }
                } else {
                    // If the audio is not stereo, we can just use the first channel:
                    audio = audioData.getChannelData(0);
                }

                webWorker.postMessage({
                    type: "transcribe",
                    requestId,
                    audio,
                    model,
                    dtype,
                    gpu,
                    diarize,
                    subtask: !model.endsWith(".en") ? subtask : null,
                    language:
                        !model.endsWith(".en") && language !== "auto"
                            ? language
                            : null,
                });
            }
        },
        [webWorker, model, dtype, gpu, subtask, language, diarize],
    );

    const transcriber = useMemo(() => {
        return {
            onInputChange,
            isBusy,
            isModelLoading,
            progressItems,
            start: postRequest,
            output: transcript,
            model,
            setModel,
            dtype,
            setDtype,
            gpu,
            setGPU,
            subtask,
            setSubtask,
            language,
            setLanguage,
            diarize,
            setDiarize,
            diarizationStatus,
            diarizationProgress,
            diarizationError,
            cancelDiarization,
        };
    }, [
        onInputChange,
        isBusy,
        isModelLoading,
        progressItems,
        postRequest,
        transcript,
        model,
        dtype,
        gpu,
        subtask,
        language,
        diarize,
        diarizationStatus,
        diarizationProgress,
        diarizationError,
        cancelDiarization,
    ]);

    return transcriber;
}
