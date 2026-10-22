
import argparse
import json
from pathlib import Path

import numpy as np


FPS = 30

FACIAL_CHANNELS = [
    "mouth_open",
    "mouth_width",
    "brow_raise",
    "head_yaw",
    "head_pitch",
    "head_roll",
    "gesture_intensity",
    "lip_closure",
    "lip_rounding",
    "lip_stretch",
    "lower_lip_contact",
]

VISEME_TARGETS = {
    "rest": (0.0, 0.0, 0.0, 0.0),
    "closed_lips": (1.0, 0.0, 0.0, 0.0),
    "teeth_lip": (0.15, 0.1, 0.0, 1.0),
    "rounded": (0.0, 1.0, 0.0, 0.0),
    "wide": (0.0, 0.0, 1.0, 0.0),
    "open": (0.0, 0.1, 0.25, 0.0),
    "tongue": (0.0, 0.0, 0.2, 0.0),
    "consonant": (0.1, 0.0, 0.1, 0.0),
}


def smooth_bidirectional(
    values: np.ndarray,
    seconds: float = 0.045,
) -> np.ndarray:
    if values.size <= 1:
        return values.copy()

    alpha = 1.0 - np.exp(
        -1.0 / (FPS * seconds)
    )

    forward = values.astype(
        np.float64,
        copy=True,
    )

    for index in range(1, len(forward)):
        forward[index] = (
            forward[index - 1]
            + alpha * (
                values[index] - forward[index - 1]
            )
        )

    backward = forward.copy()

    for index in range(
        len(backward) - 2,
        -1,
        -1,
    ):
        backward[index] = (
            backward[index + 1]
            + alpha * (
                forward[index] - backward[index + 1]
            )
        )

    return backward


def build_facial_targets(
    segments: list[dict],
    frame_count: int,
) -> np.ndarray:
    times = np.arange(
        frame_count,
        dtype=np.float64,
    ) / FPS

    accumulated = np.zeros(
        (frame_count, 4),
        dtype=np.float64,
    )

    weights = np.zeros(
        frame_count,
        dtype=np.float64,
    )

    for segment in segments:
        category = segment.get("viseme", "rest")

        if category not in VISEME_TARGETS:
            continue

        try:
            start = float(segment["start"])
            end = float(segment["end"])
        except (KeyError, TypeError, ValueError):
            continue

        if (
            not np.isfinite(start)
            or not np.isfinite(end)
            or end <= start
        ):
            continue

        center = (start + end) / 2.0

        spread = max(
            0.045,
            (end - start) * 0.45,
        )

        distance = (
            times - center
        ) / spread

        influence = np.exp(
            -0.5 * distance * distance
        )

        influence[
            np.abs(distance) > 3.0
        ] = 0.0

        target = np.asarray(
            VISEME_TARGETS[category],
            dtype=np.float64,
        )

        accumulated += (
            influence[:, None] * target
        )

        weights += influence

    targets = np.zeros_like(accumulated)

    active = weights > 1e-8

    targets[active] = (
        accumulated[active]
        / weights[active, None]
    )

    confidence = np.clip(
        weights,
        0.0,
        1.0,
    )

    targets *= confidence[:, None]

    return np.clip(
        targets,
        0.0,
        1.0,
    )


def build_pause_mask(
    pauses: list[dict],
    frame_count: int,
) -> np.ndarray:
    times = np.arange(
        frame_count,
        dtype=np.float64,
    ) / FPS

    mask = np.zeros(
        frame_count,
        dtype=np.float64,
    )

    for pause in pauses:
        try:
            start = float(pause["start"])
            end = float(pause["end"])
        except (KeyError, TypeError, ValueError):
            continue

        if (
            not np.isfinite(start)
            or not np.isfinite(end)
            or end - start < 0.18
        ):
            continue

        fade = min(
            0.07,
            (end - start) * 0.25,
        )

        if fade <= 0:
            continue

        fade_in = np.clip(
            (times - start) / fade,
            0.0,
            1.0,
        )

        fade_out = np.clip(
            (end - times) / fade,
            0.0,
            1.0,
        )

        influence = np.minimum(
            fade_in,
            fade_out,
        )

        influence = (
            influence
            * influence
            * (3.0 - 2.0 * influence)
        )

        inside = (
            (times >= start)
            & (times <= end)
        )

        influence = np.where(
            inside,
            influence,
            0.0,
        )

        mask = np.maximum(
            mask,
            influence,
        )

    return mask


def generate_facial_motion(
    motion_path: Path,
    viseme_path: Path,
    output_path: Path,
) -> dict:
    if not motion_path.is_file():
        raise FileNotFoundError(
            f"Motion not found: {motion_path}"
        )

    if not viseme_path.is_file():
        raise FileNotFoundError(
            f"Visemes not found: {viseme_path}"
        )

    if output_path.resolve() == motion_path.resolve():
        raise ValueError(
            "Output cannot overwrite input motion"
        )

    motion = np.load(
        motion_path,
        allow_pickle=False,
    )

    if (
        motion.ndim != 2
        or motion.shape[1] != 7
        or motion.shape[0] == 0
    ):
        raise ValueError(
            "Input motion must have seven channels"
        )

    if not np.isfinite(motion).all():
        raise ValueError(
            "Motion contains invalid values"
        )

    data = json.loads(
        viseme_path.read_text(
            encoding="utf-8"
        )
    )

    segments = data.get("segments", [])
    pauses = data.get("pauses", [])

    if not isinstance(segments, list):
        raise ValueError(
            "Segments must be a list"
        )

    if not isinstance(pauses, list):
        raise ValueError(
            "Pauses must be a list"
        )

    frame_count = motion.shape[0]

    targets = build_facial_targets(
        segments,
        frame_count,
    )

    pause_mask = build_pause_mask(
        pauses,
        frame_count,
    )

    targets *= (
        1.0 - pause_mask[:, None]
    )

    smoothed = np.column_stack(
        [
            smooth_bidirectional(
                targets[:, index]
            )
            for index in range(4)
        ]
    )

    smoothed = np.clip(
        smoothed,
        0.0,
        1.0,
    )

    result = np.concatenate(
        (
            motion.astype(np.float32),
            smoothed.astype(np.float32),
        ),
        axis=1,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.save(
        output_path,
        result,
    )

    channel_statistics = {}

    for index, name in enumerate(
        FACIAL_CHANNELS[7:],
        start=7,
    ):
        values = result[:, index]

        channel_statistics[name] = {
            "mean": round(
                float(values.mean()),
                4,
            ),
            "maximum": round(
                float(values.max()),
                4,
            ),
        }

    return {
        "output_file": str(output_path),
        "frames": int(result.shape[0]),
        "fps": FPS,
        "duration": round(
            result.shape[0] / FPS,
            3,
        ),
        "dimensions": int(result.shape[1]),
        "channels": FACIAL_CHANNELS,
        "viseme_segments": len(segments),
        "detected_pauses": len(pauses),
        "original_channels_preserved": bool(
            np.array_equal(
                result[:, :7],
                motion,
            )
        ),
        "facial_statistics": channel_statistics,
    }


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--motion",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--visemes",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    result = generate_facial_motion(
        args.motion,
        args.visemes,
        args.output,
    )

    print(
        json.dumps(
            result,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
