
import argparse
import json
from pathlib import Path

import numpy as np


FPS = 30

VISEME_VALUES = {
    "rest": (0.0, 0.0),
    "closed_lips": (0.02, 0.15),
    "teeth_lip": (0.16, 0.35),
    "rounded": (0.42, 0.2),
    "wide": (0.38, 0.85),
    "open": (0.9, 0.55),
    "tongue": (0.25, 0.45),
    "consonant": (0.18, 0.4),
}


def smooth_signal(
    values: np.ndarray,
    window: int = 5,
) -> np.ndarray:
    if window <= 1:
        return values.copy()

    kernel = np.ones(window, dtype=np.float32)
    kernel /= kernel.sum()

    padding = window // 2

    padded = np.pad(
        values,
        (padding, padding),
        mode="edge",
    )

    return np.convolve(
        padded,
        kernel,
        mode="valid",
    )


def generate_viseme_motion(
    baseline_path: Path,
    viseme_path: Path,
    output_path: Path,
) -> dict:
    baseline = np.load(
        baseline_path,
        allow_pickle=False,
    )

    if baseline.ndim != 2 or baseline.shape[1] != 7:
        raise ValueError(
            "Baseline motion must have 7 dimensions"
        )

    if not np.isfinite(baseline).all():
        raise ValueError(
            "Baseline contains invalid values"
        )

    viseme_data = json.loads(
        viseme_path.read_text(encoding="utf-8")
    )

    segments = viseme_data.get("segments", [])

    frame_count = baseline.shape[0]

    motion = baseline.astype(
        np.float32,
        copy=True,
    )

    mouth_open = np.zeros(
        frame_count,
        dtype=np.float32,
    )

    mouth_width = np.zeros(
        frame_count,
        dtype=np.float32,
    )

    weights = np.zeros(
        frame_count,
        dtype=np.float32,
    )

    times = np.arange(
        frame_count,
        dtype=np.float32,
    ) / FPS

    for segment in segments:
        category = segment.get("viseme", "rest")

        if category not in VISEME_VALUES:
            continue

        start = float(segment["start"])
        end = float(segment["end"])

        if end <= start:
            continue

        open_value, width_value = VISEME_VALUES[
            category
        ]

        center = (start + end) / 2

        spread = max(
            0.055,
            (end - start) / 2,
        )

        influence = np.exp(
            -0.5 * ((times - center) / spread) ** 2
        ).astype(np.float32)

        mouth_open += influence * open_value
        mouth_width += influence * width_value
        weights += influence

    active = weights > 0.01

    target_open = np.zeros(
        frame_count,
        dtype=np.float32,
    )

    target_width = np.zeros(
        frame_count,
        dtype=np.float32,
    )

    target_open[active] = (
        mouth_open[active] / weights[active]
    )

    target_width[active] = (
        mouth_width[active] / weights[active]
    )

    baseline_open = np.clip(
        baseline[:, 0],
        0,
        1,
    )

    baseline_width = np.clip(
        baseline[:, 1],
        0,
        1,
    )

    confidence = np.clip(
        weights,
        0,
        1,
    )

    blended_open = (
        (1 - confidence) * baseline_open
        + confidence * target_open
    )

    blended_width = (
        (1 - confidence) * baseline_width
        + confidence * target_width
    )

    motion[:, 0] = np.clip(
        smooth_signal(blended_open),
        0,
        1,
    )

    motion[:, 1] = np.clip(
        smooth_signal(blended_width),
        0,
        1,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.save(output_path, motion)

    return {
        "output_file": str(output_path),
        "frames": int(motion.shape[0]),
        "fps": FPS,
        "dimensions": int(motion.shape[1]),
        "viseme_segments": len(segments),
        "duration": round(frame_count / FPS, 2),
        "mouth_open_min": round(
            float(motion[:, 0].min()),
            4,
        ),
        "mouth_open_max": round(
            float(motion[:, 0].max()),
            4,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--baseline",
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

    result = generate_viseme_motion(
        args.baseline,
        args.visemes,
        args.output,
    )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
