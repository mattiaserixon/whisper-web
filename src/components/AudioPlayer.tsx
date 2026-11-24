import { useEffect, useRef } from "react";

export interface AudioSeekRequest {
    time: number;
    nonce: number;
}

export default function AudioPlayer(props: {
    audioUrl: string;
    mimeType: string;
    seekRequest?: AudioSeekRequest;
    onTimeUpdate?: (currentTime: number) => void;
    onPlayStateChange?: (isPlaying: boolean) => void;
}) {
    const audioPlayer = useRef<HTMLAudioElement>(null);
    const audioSource = useRef<HTMLSourceElement>(null);

    // Updates src when url changes
    useEffect(() => {
        if (audioPlayer.current && audioSource.current) {
            audioSource.current.src = props.audioUrl;
            audioPlayer.current.load();
            props.onTimeUpdate?.(0);
            props.onPlayStateChange?.(false);
        }
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [props.audioUrl]);

    // Seek to requested timestamp without auto-playing if currently paused
    useEffect(() => {
        if (!props.seekRequest || !audioPlayer.current) return;
        const player = audioPlayer.current;
        const maxDuration = Number.isFinite(player.duration)
            ? player.duration
            : undefined;
        const target =
            maxDuration !== undefined
                ? Math.min(Math.max(0, props.seekRequest.time), maxDuration)
                : Math.max(0, props.seekRequest.time);

        player.currentTime = target;
        props.onTimeUpdate?.(target);
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [props.seekRequest]);

    return (
        <div className='flex relative z-10 p-4 w-full mt-4'>
            <audio
                ref={audioPlayer}
                controls
                onTimeUpdate={(e) =>
                    props.onTimeUpdate?.(e.currentTarget.currentTime)
                }
                onSeeked={(e) =>
                    props.onTimeUpdate?.(e.currentTarget.currentTime)
                }
                onPlay={() => props.onPlayStateChange?.(true)}
                onPause={() => props.onPlayStateChange?.(false)}
                onEnded={() => props.onPlayStateChange?.(false)}
                className='w-full h-14 rounded-lg bg-white shadow-xl shadow-black/5 ring-1 ring-slate-700/10'
            >
                <source ref={audioSource} type={props.mimeType}></source>
            </audio>
        </div>
    );
}
