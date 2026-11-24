const HOURS_IN_SECONDS = 3600;
const MINUTES_IN_SECONDS = 60;

function padTime(time: number, padding: number = 2) {
    return String(time).padStart(padding, "0");
}

function splitTotalMilliseconds(timeSeconds: number) {
    const clampedMs = Math.max(0, Math.round(timeSeconds * 1000));
    const totalSeconds = Math.floor(clampedMs / 1000);
    const milliseconds = clampedMs % 1000;

    const hours = Math.floor(totalSeconds / HOURS_IN_SECONDS);
    const minutes = Math.floor(
        (totalSeconds % HOURS_IN_SECONDS) / MINUTES_IN_SECONDS,
    );
    const seconds = totalSeconds % MINUTES_IN_SECONDS;

    return { hours, minutes, seconds, milliseconds };
}

function formatClockSeconds(totalWholeSeconds: number, forceHours = false) {
    const safeSeconds = Math.max(0, Math.floor(totalWholeSeconds));
    const hours = Math.floor(safeSeconds / HOURS_IN_SECONDS);
    const minutes = Math.floor(
        (safeSeconds % HOURS_IN_SECONDS) / MINUTES_IN_SECONDS,
    );
    const seconds = safeSeconds % MINUTES_IN_SECONDS;

    const hoursString =
        forceHours || hours > 0 ? padTime(hours) + ":" : "";
    const minutesString = padTime(minutes) + ":";
    const secondsString = padTime(seconds);

    return `${hoursString}${minutesString}${secondsString}`;
}

export function formatAudioTimestamp(
    time: number,
    forceHours: boolean = false,
) {
    return formatClockSeconds(Math.floor(time), forceHours);
}

/**
 * Formats a segment's [start, end] range for human reading:
 * - Shows `mm:ss–mm:ss` (or `hh:mm:ss` when recording/segment reaches 1 hour).
 * - Shows start time alone if `end` is null/unknown.
 * - Returns null if `start` is null/unknown (never turns null into fake 00:00).
 */
export function formatDisplayTimeRange(
    start: number | null,
    end: number | null,
    forceHours: boolean = false,
): string | null {
    if (start === null || Number.isNaN(start)) {
        return null;
    }
    const showHours =
        forceHours || start >= HOURS_IN_SECONDS || (end ?? 0) >= HOURS_IN_SECONDS;
    const startWhole = Math.floor(Math.max(0, start));
    const startStr = formatClockSeconds(startWhole, showHours);

    if (end === null || Number.isNaN(end) || end <= start) {
        return startStr;
    }

    const endWhole = Math.max(startWhole + 1, Math.round(end));
    const endStr = formatClockSeconds(endWhole, showHours);
    return `${startStr}–${endStr}`;
}

export function formatSrtTimestamp(time: number) {
    const { hours, minutes, seconds, milliseconds } =
        splitTotalMilliseconds(time);

    const hoursString = padTime(hours) + ":";
    const minutesString = padTime(minutes) + ":";
    const secondsString = padTime(seconds) + ",";
    const millisecondsString = padTime(milliseconds, 3);

    return `${hoursString}${minutesString}${secondsString}${millisecondsString}`;
}

export function formatSrtTimeRange(start: number, end: number) {
    const safeStart = Math.max(0, start);
    const safeEnd = Math.max(safeStart + 0.001, end);
    return `${formatSrtTimestamp(safeStart)} --> ${formatSrtTimestamp(safeEnd)}`;
}

/**
 * Resolves a defensible end boundary for a segment (e.g., for SRT cues or active playback
 * highlighting) when Whisper left the trailing end timestamp null.
 */
export function resolveSegmentEndSeconds(
    startSeconds: number,
    endSeconds: number | null,
    nextStartSeconds: number | null,
    audioDurationSeconds: number | null,
    text: string,
): { endSeconds: number; isEstimated: boolean } {
    if (endSeconds !== null && endSeconds > startSeconds) {
        return { endSeconds, isEstimated: false };
    }
    if (nextStartSeconds !== null && nextStartSeconds > startSeconds) {
        return { endSeconds: nextStartSeconds, isEstimated: false };
    }
    if (
        audioDurationSeconds !== null &&
        audioDurationSeconds > startSeconds
    ) {
        return { endSeconds: audioDurationSeconds, isEstimated: false };
    }
    const wordCount = text.trim().split(/\s+/).filter(Boolean).length;
    const estimatedDuration = Math.max(1.0, Math.min(5.0, wordCount * 0.35));
    return {
        endSeconds: startSeconds + estimatedDuration,
        isEstimated: true,
    };
}
