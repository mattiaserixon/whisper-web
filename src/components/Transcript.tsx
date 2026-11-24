import { useEffect, useMemo, useRef } from "react";
import { useTranslation } from "react-i18next";

import { TranscriberData } from "../hooks/useTranscriber";
import {
    formatAudioTimestamp,
    formatDisplayTimeRange,
    resolveSegmentEndSeconds,
} from "../utils/AudioUtils";
import {
    buildJsonExport,
    buildPlainTxtExport,
    buildSrtExport,
    buildStructuredTxtExport,
    formatSpeakerAttribution,
    SpeakerLabelTranslations,
} from "../utils/TranscriptExport";
import {
    DiarizationStatus,
    SpeakerAttribution,
} from "../utils/TranscriptTypes";

interface Props {
    transcribedData: TranscriberData | undefined;
    diarizeEnabled?: boolean;
    diarizationStatus?: DiarizationStatus;
    diarizationProgress?: number;
    onCancelDiarization?: () => void;
    canSeek?: boolean;
    currentAudioTime?: number;
    isAudioPlaying?: boolean;
    onSeek?: (seconds: number) => void;
}

const SPEAKER_PALETTE = [
    "bg-indigo-50 text-indigo-700 ring-indigo-600/20",
    "bg-emerald-50 text-emerald-700 ring-emerald-600/20",
    "bg-amber-50 text-amber-800 ring-amber-600/20",
    "bg-rose-50 text-rose-700 ring-rose-600/20",
    "bg-sky-50 text-sky-700 ring-sky-600/20",
    "bg-purple-50 text-purple-700 ring-purple-600/20",
];

function getSpeakerBadgeClasses(speaker: SpeakerAttribution): string {
    if (
        speaker.kind === "speaker" &&
        speaker.speakerIndex !== undefined &&
        speaker.speakerIndex >= 1
    ) {
        return SPEAKER_PALETTE[
            (speaker.speakerIndex - 1) % SPEAKER_PALETTE.length
        ];
    }
    if (speaker.kind === "overlap" || speaker.kind === "mixed") {
        return "bg-violet-50 text-violet-700 ring-violet-600/20";
    }
    return "bg-slate-100 text-slate-600 ring-slate-500/20";
}

export default function Transcript({
    transcribedData,
    diarizeEnabled = false,
    diarizationStatus = "disabled",
    diarizationProgress = 0,
    onCancelDiarization,
    canSeek = false,
    currentAudioTime = 0,
    isAudioPlaying = false,
    onSeek,
}: Props) {
    const { t } = useTranslation();
    const endOfMessagesRef = useRef<HTMLDivElement>(null);
    const shouldAutoScrollRef = useRef<boolean>(true);

    const segments = useMemo(
        () => transcribedData?.chunks ?? [],
        [transcribedData?.chunks],
    );

    const includeHours = useMemo(
        () =>
            segments.some(
                (s) =>
                    (s.startSeconds !== null && s.startSeconds >= 3600) ||
                    (s.endSeconds !== null && s.endSeconds >= 3600),
            ),
        [segments],
    );

    const isDiarizationActive =
        diarizationStatus === "loading_models" ||
        diarizationStatus === "analyzing" ||
        diarizationStatus === "pending";

    const hasAnySpeakerLabels = useMemo(
        () => segments.some((seg) => seg.speaker !== null),
        [segments],
    );

    const showSpeakerColumn =
        hasAnySpeakerLabels || (diarizeEnabled && isDiarizationActive);

    const speakerLabels: SpeakerLabelTranslations = useMemo(
        () => ({
            speakerN: (n: number) => t("transcript.speaker_n", { number: n }),
            overlappingSpeech: t("transcript.overlapping_speech"),
            multipleSpeakers: t("transcript.multiple_speakers"),
            unknownSpeaker: t("transcript.unknown_speaker"),
        }),
        [t],
    );

    // Track whether the user has scrolled away from the bottom so live streaming doesn't yank scroll position
    useEffect(() => {
        const handleWindowScroll = () => {
            const distanceFromBottom =
                document.documentElement.scrollHeight -
                (window.scrollY + window.innerHeight);
            shouldAutoScrollRef.current = distanceFromBottom < 180;
        };
        window.addEventListener("scroll", handleWindowScroll, {
            passive: true,
        });
        return () => window.removeEventListener("scroll", handleWindowScroll);
    }, []);

    useEffect(() => {
        if (transcribedData?.isBusy && shouldAutoScrollRef.current) {
            endOfMessagesRef.current?.scrollIntoView({ behavior: "auto" });
        }
    }, [segments, transcribedData?.isBusy]);

    const activeSegmentIndex = useMemo(() => {
        if (!canSeek || (!isAudioPlaying && currentAudioTime <= 0)) {
            return -1;
        }
        for (let i = 0; i < segments.length; i++) {
            const seg = segments[i];
            if (seg.startSeconds === null) continue;

            let nextStart: number | null = null;
            for (let j = i + 1; j < segments.length; j++) {
                if (segments[j].startSeconds !== null) {
                    nextStart = segments[j].startSeconds;
                    break;
                }
            }

            const { endSeconds: effectiveEnd } = resolveSegmentEndSeconds(
                seg.startSeconds,
                seg.endSeconds,
                nextStart,
                transcribedData?.audioDuration ?? null,
                seg.text,
            );

            if (
                currentAudioTime >= seg.startSeconds &&
                currentAudioTime < effectiveEnd
            ) {
                return i;
            }
        }
        return -1;
    }, [
        canSeek,
        isAudioPlaying,
        currentAudioTime,
        segments,
        transcribedData?.audioDuration,
    ]);

    const saveBlob = (blob: Blob, filename: string) => {
        const url = URL.createObjectURL(blob);
        const link = document.createElement("a");
        link.href = url;
        link.download = filename;
        link.click();
        URL.revokeObjectURL(url);
    };

    const exportPlainTXT = () => {
        const text = buildPlainTxtExport(segments);
        saveBlob(
            new Blob([text], { type: "text/plain;charset=utf-8" }),
            "transcript.txt",
        );
    };

    const exportStructuredTXT = () => {
        const text = buildStructuredTxtExport(segments, speakerLabels);
        saveBlob(
            new Blob([text], { type: "text/plain;charset=utf-8" }),
            "transcript-structured.txt",
        );
    };

    const exportJSON = () => {
        const jsonData = buildJsonExport(segments, speakerLabels);
        saveBlob(
            new Blob([jsonData], { type: "application/json;charset=utf-8" }),
            "transcript.json",
        );
    };

    const exportSRT = () => {
        const srt = buildSrtExport(
            segments,
            speakerLabels,
            transcribedData?.audioDuration,
        );
        saveBlob(
            new Blob([srt], { type: "text/plain;charset=utf-8" }),
            "transcript.srt",
        );
    };

    const exportButtons = [
        { name: t("transcript.export_txt_plain"), onClick: exportPlainTXT },
        {
            name: t("transcript.export_txt_structured"),
            onClick: exportStructuredTXT,
        },
        { name: "JSON", onClick: exportJSON },
        { name: "SRT", onClick: exportSRT },
    ];

    if (!transcribedData || segments.length === 0) {
        return null;
    }

    const desktopGridCols = showSpeakerColumn
        ? "sm:grid-cols-[9.5rem_10rem_minmax(0,1fr)]"
        : "sm:grid-cols-[9.5rem_minmax(0,1fr)]";

    return (
        <div className='w-full flex flex-col mt-2 p-4'>
            {!transcribedData.isBusy &&
                (diarizationStatus === "loading_models" ||
                    diarizationStatus === "analyzing") && (
                    <div
                        role='status'
                        aria-live='polite'
                        className='mb-3 flex flex-col gap-2 rounded-xl bg-indigo-50/90 px-4 py-3 text-sm text-indigo-950 ring-1 ring-indigo-200'
                    >
                        <div className='flex items-center justify-between gap-3'>
                            <div className='flex items-center gap-2.5'>
                                <span className='inline-block h-4 w-4 animate-spin rounded-full border-2 border-indigo-600 border-t-transparent' />
                                <span>
                                    {diarizationStatus === "loading_models"
                                        ? t("transcript.diarization_loading")
                                        : t(
                                              "transcript.diarization_analyzing",
                                              {
                                                  progress: diarizationProgress,
                                              },
                                          )}
                                </span>
                            </div>
                            {onCancelDiarization && (
                                <button
                                    type='button'
                                    onClick={onCancelDiarization}
                                    className='shrink-0 rounded-md bg-white px-2.5 py-1 text-xs font-medium text-slate-700 shadow-xs ring-1 ring-slate-300 hover:bg-slate-50'
                                >
                                    {t("transcript.cancel_diarization")}
                                </button>
                            )}
                        </div>
                        {diarizationStatus === "analyzing" && (
                            <div className='h-1.5 w-full overflow-hidden rounded-full bg-indigo-200/70'>
                                <div
                                    className='h-full rounded-full bg-indigo-600 transition-all duration-150'
                                    style={{
                                        width: `${Math.max(4, diarizationProgress)}%`,
                                    }}
                                />
                            </div>
                        )}
                    </div>
                )}

            {!transcribedData.isBusy && diarizationStatus === "unavailable" && (
                <div
                    role='status'
                    className='mb-3 rounded-xl bg-amber-50 px-4 py-2.5 text-sm text-amber-900 ring-1 ring-amber-200'
                >
                    {t("transcript.diarization_unavailable")}
                </div>
            )}

            <div className='w-full overflow-hidden rounded-xl bg-white shadow-xl shadow-black/5 ring-1 ring-slate-700/10'>
                <div
                    className={`hidden sm:grid ${desktopGridCols} gap-3 border-b border-slate-200 bg-slate-50/90 px-4 py-2.5 text-xs font-semibold uppercase tracking-wide text-slate-500`}
                >
                    <div>{t("transcript.column_time")}</div>
                    {showSpeakerColumn && (
                        <div>{t("transcript.column_speaker")}</div>
                    )}
                    <div>{t("transcript.column_text")}</div>
                </div>

                <div className='divide-y divide-slate-100'>
                    {segments.map((seg, i) => {
                        const isActive = i === activeSegmentIndex;
                        const timeRangeLabel =
                            formatDisplayTimeRange(
                                seg.startSeconds,
                                seg.endSeconds,
                                includeHours,
                            ) ?? "—";
                        const speakerLabel = formatSpeakerAttribution(
                            seg.speaker,
                            speakerLabels,
                        );
                        const isSeekable =
                            canSeek &&
                            onSeek !== undefined &&
                            seg.startSeconds !== null;

                        return (
                            <div
                                key={seg.id || `${i}-${seg.startSeconds ?? "u"}`}
                                data-testid='transcript-segment'
                                data-active={isActive ? "true" : "false"}
                                className={`flex flex-col gap-1.5 sm:grid ${desktopGridCols} sm:items-baseline sm:gap-3 px-4 py-3 transition-colors ${
                                    isActive
                                        ? "bg-indigo-50/80 shadow-[inset_4px_0_0_0_rgb(79,70,229)]"
                                        : transcribedData.isBusy
                                          ? "bg-slate-50/70"
                                          : "bg-white hover:bg-slate-50/60"
                                }`}
                            >
                                <div className='flex flex-wrap items-center gap-2 sm:contents'>
                                    <div className='shrink-0'>
                                        {isSeekable &&
                                        seg.startSeconds !== null ? (
                                            <button
                                                type='button'
                                                onClick={() =>
                                                    onSeek(seg.startSeconds!)
                                                }
                                                aria-label={t(
                                                    "transcript.seek_to_time",
                                                    {
                                                        time: formatAudioTimestamp(
                                                            seg.startSeconds,
                                                            includeHours,
                                                        ),
                                                    },
                                                )}
                                                className='inline-flex items-center gap-1 rounded-md bg-slate-100 px-2 py-0.5 font-mono text-xs font-medium tabular-nums text-slate-700 hover:bg-indigo-100 hover:text-indigo-800 focus:outline-none focus:ring-2 focus:ring-indigo-500 cursor-pointer transition-colors'
                                            >
                                                <span>{timeRangeLabel}</span>
                                            </button>
                                        ) : (
                                            <span className='inline-flex items-center rounded-md bg-slate-100 px-2 py-0.5 font-mono text-xs tabular-nums text-slate-500'>
                                                {timeRangeLabel}
                                            </span>
                                        )}
                                    </div>

                                    {showSpeakerColumn && (
                                        <div className='shrink-0'>
                                            {seg.speaker && speakerLabel ? (
                                                <span
                                                    data-testid='speaker-badge'
                                                    className={`inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-medium ring-1 ring-inset ${getSpeakerBadgeClasses(
                                                        seg.speaker,
                                                    )}`}
                                                >
                                                    {speakerLabel}
                                                </span>
                                            ) : isDiarizationActive ? (
                                                <span className='inline-flex items-center rounded-full bg-slate-100 px-2.5 py-0.5 text-xs text-slate-400 animate-pulse'>
                                                    {t(
                                                        "transcript.analyzing_speaker_badge",
                                                    )}
                                                </span>
                                            ) : (
                                                <span className='text-xs text-slate-400'>
                                                    —
                                                </span>
                                            )}
                                        </div>
                                    )}
                                </div>

                                <div className='text-sm sm:text-base leading-relaxed text-slate-800'>
                                    {seg.text.trim()}
                                </div>
                            </div>
                        );
                    })}
                </div>
            </div>

            {!transcribedData.isBusy && (
                <div className='w-full flex flex-wrap justify-center gap-2 mt-4'>
                    {exportButtons.map((button) => (
                        <button
                            key={button.name}
                            type='button'
                            onClick={button.onClick}
                            className='text-white bg-green-600 hover:bg-green-700 focus:ring-4 focus:ring-green-300 font-medium rounded-lg text-sm px-4 py-2 text-center inline-flex items-center cursor-pointer transition-colors'
                        >
                            {t("transcript.export")} {button.name}
                        </button>
                    ))}
                </div>
            )}

            {transcribedData.tps !== undefined && (
                <p className='text-sm text-center mt-4'>
                    <span className='font-semibold text-black'>
                        {transcribedData.tps.toFixed(2)}
                    </span>{" "}
                    <span className='text-gray-500'>
                        {t("transcript.tokens_per_second")}
                    </span>
                </p>
            )}
            <div ref={endOfMessagesRef} />
        </div>
    );
}
