
import argparse
import json
import shutil
import subprocess
import wave
from pathlib import Path

import numpy as np


FPS = 30
SAMPLE_RATE = 16000


def get_wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as audio:
        if audio.getnframes() == 0:
            raise ValueError("Audio file is empty")

        return audio.getnframes() / audio.getframerate()


def refine_voice(
    input_path: Path,
    output_path: Path,
    speed: float = 0.9,
) -> dict:
    if not 0.75 <= speed <= 1.0:
        raise ValueError(
            "Speed must be between 0.75 and 1.0"
        )

    if not input_path.is_file():
        raise FileNotFoundError(
            f"Audio not found: {input_path}"
        )

    if shutil.which("ffmpeg") is None:
        raise RuntimeError("FFmpeg is not installed")

    if input_path.resolve() == output_path.resolve():
        raise ValueError(
            "Input and output audio must be different"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(input_path),
        "-vn",
        "-af",
        f"atempo={speed}",
        "-ar",
        str(SAMPLE_RATE),
        "-ac",
        "1",
        "-c:a",
        "pcm_s16le",
        str(output_path),
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        output_path.unlink(missing_ok=True)

        raise RuntimeError(
            result.stderr.strip()
            or "Audio refinement failed"
        )

    return {
        "input_file": input_path.name,
        "output_file": output_path.name,
        "speed": speed,
        "original_duration": round(
            get_wav_duration(input_path),
            3,
        ),
        "refined_duration": round(
            get_wav_duration(output_path),
            3,
        ),
        "sample_rate": SAMPLE_RATE,
        "pitch_preserving": True,
    }


def retime_motion(
    input_path: Path,
    output_path: Path,
    target_duration: float,
) -> dict:
    if not input_path.is_file():
        raise FileNotFoundError(
            f"Motion not found: {input_path}"
        )

    if target_duration <= 0:
        raise ValueError(
            "Target duration must be positive"
        )

    if input_path.resolve() == output_path.resolve():
        raise ValueError(
            "Input and output motion must be different"
        )

    motion = np.load(
        input_path,
        allow_pickle=False,
    )

    if (
        motion.ndim != 2
        or motion.shape[1] != 7
        or motion.shape[0] < 2
        or not np.isfinite(motion).all()
    ):
        raise ValueError(
            "Invalid seven-channel motion data"
        )

    original_frames = motion.shape[0]

    target_frames = max(
        2,
        round(target_duration * FPS),
    )

    source_positions = np.linspace(
        0.0,
        1.0,
        original_frames,
    )

    target_positions = np.linspace(
        0.0,
        1.0,
        target_frames,
    )

    refined = np.column_stack(
        [
            np.interp(
                target_positions,
                source_positions,
                motion[:, channel],
            )
            for channel in range(7)
        ]
    ).astype(np.float32)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.save(output_path, refined)

    return {
        "input_file": input_path.name,
        "output_file": output_path.name,
        "original_frames": original_frames,
        "refined_frames": target_frames,
        "fps": FPS,
        "duration": round(
            target_frames / FPS,
            3,
        ),
        "dimensions": 7,
    }


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--audio",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-audio",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--motion",
        type=Path,
        nargs="+",
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--speed",
        type=float,
        default=0.9,
    )

    args = parser.parse_args()

    audio_result = refine_voice(
        args.audio,
        args.output_audio,
        args.speed,
    )

    duration = audio_result["refined_duration"]

    motion_results = []

    for motion_path in args.motion:
        output_path = (
            args.output_dir
            / f"{motion_path.stem}_slow.npy"
        )

        motion_results.append(
            retime_motion(
                motion_path,
                output_path,
                duration,
            )
        )

    print(
        json.dumps(
            {
                "audio": audio_result,
                "motion": motion_results,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
