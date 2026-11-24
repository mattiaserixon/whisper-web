export type SpeakerKind = "speaker" | "overlap" | "mixed" | "unknown";

export interface SpeakerAttribution {
    kind: SpeakerKind;
    /** 1-based speaker index within the current recording when kind === "speaker" */
    speakerIndex?: number;
    /** 1-based speaker indices when kind === "overlap" or "mixed" */
    speakerIndices?: number[];
}

export type DiarizationStatus =
    | "disabled"
    | "pending"
    | "loading_models"
    | "analyzing"
    | "ready"
    | "cancelled"
    | "unavailable";

export interface TranscriptSegment {
    id: string;
    text: string;
    startSeconds: number | null;
    endSeconds: number | null;
    /** Preserved [start, end] tuple for backwards-compatible JSON consumers */
    timestamp: [number | null, number | null];
    speaker?: SpeakerAttribution | null;
    finalised?: boolean;
}
