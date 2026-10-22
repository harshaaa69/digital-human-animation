
import argparse
import json
from pathlib import Path

import numpy as np


FPS = 30

VISEME_VALUES = {
    "rest": (0.02, 0.15),
    "closed_lips": (0.0, 0.12),
    "teeth_lip": (0.12, 0.32),
    "rounded": (0.38, 0.22),
    "wide": (0.34, 0.70),
    "open": (0.72, 0.48),
    "tongue": (0.22, 0.40),
    "consonant": (0.18, 0.35),
}

MIN_TRANSITION_SECONDS = 0.09
ATTACK_SECONDS = 0.065
RELEASE_SECONDS = 0.10
MAX_CHANGE_PER_SECOND = 1.5
VISEME_STRENGTH = 0.65

MIN_PAUSE_DURATION = 0.18
PAUSE_FADE_SECONDS = 0.08
PAUSE_MOUTH_OPEN = 0.015
PAUSE_MOUTH_WIDTH = 0.12


def clamp_signal(values: np.ndarray) -> np.ndarray:
    return np.clip(values, 0.0, 1.0)


def smooth_transition(
    values: np.ndarray,
    attack_seconds: float,
    release_seconds: float,
    fps: int,
) -> np.ndarray:
    result = np.zeros_like(
        values,
        dtype=np.float64,
    )

    if len(values) == 0:
        return result

    result[0] = values[0]

    attack_alpha = 1.0 - np.exp(
        -1.0 / (fps * attack_seconds)
    )

    release_alpha = 1.0 - np.exp(
        -1.0 / (fps * release_seconds)
    )

    for index in range(1, len(values)):
        previous = result[index - 1]
        target = values[index]

        alpha = (
            attack_alpha
            if target > previous
            else release_alpha
        )

        result[index] = (
            previous + alpha * (target - previous)
        )

    return result


def limit_motion_speed(
    values: np.ndarray,
    max_change_per_second: float,
    fps: int,
) -> np.ndarray:
    result = values.copy()

    maximum_step = max_change_per_second / fps

    for index in range(1, len(result)):
        difference = (
            result[index] - result[index - 1]
        )

        difference = np.clip(
            difference,
            -maximum_step,
            maximum_step,
        )

        result[index] = (
            result[index - 1] + difference
        )

    return result


def build_viseme_targets(
    segments: list[dict],
    frame_count: int,
    fps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    times = np.arange(
        frame_count,
        dtype=np.float64,
    ) / fps

    opening_sum = np.zeros(
        frame_count,
        dtype=np.float64,
    )

    width_sum = np.zeros(
        frame_count,
        dtype=np.float64,
    )

    weights = np.zeros(
        frame_count,
        dtype=np.float64,
    )

    for segment in segments:
        category = segment.get("viseme", "rest")

        if category not in VISEME_VALUES:
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

        center = (start + end) / 2

        duration = max(
            end - start,
            MIN_TRANSITION_SECONDS,
        )

        spread = max(
            0.045,
            duration * 0.65,
        )

        influence = np.exp(
            -0.5
            * ((times - center) / spread) ** 2
        )

        opening, width = VISEME_VALUES[category]

        opening_sum += influence * opening
        width_sum += influence * width
        weights += influence

    target_open = np.zeros(
        frame_count,
        dtype=np.float64,
    )

    target_width = np.zeros(
        frame_count,
        dtype=np.float64,
    )

    active = weights > 1e-8

    target_open[active] = (
        opening_sum[active] / weights[active]
    )

    target_width[active] = (
        width_sum[active] / weights[active]
    )

    confidence = np.clip(
        weights,
        0.0,
        1.0,
    )

    return target_open, target_width, confidence


def build_pause_envelope(
    pauses: list[dict],
    frame_count: int,
    fps: int,
) -> tuple[np.ndarray, int]:
    times = np.arange(
        frame_count,
        dtype=np.float64,
    ) / fps

    envelope = np.zeros(
        frame_count,
        dtype=np.float64,
    )

    applied_pauses = 0

    for pause in pauses:
        try:
            start = float(pause["start"])
            end = float(pause["end"])
        except (KeyError, TypeError, ValueError):
            continue

        if (
            not np.isfinite(start)
            or not np.isfinite(end)
            or end - start < MIN_PAUSE_DURATION
        ):
            continue

        duration = end - start

        fade = min(
            PAUSE_FADE_SECONDS,
            duration * 0.3,
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

        shape = np.minimum(
            fade_in,
            fade_out,
        )

        shape = (
            shape * shape * (3.0 - 2.0 * shape)
        )

        inside = (
            (times >= start)
            & (times <= end)
        )

        shape = np.where(
            inside,
            shape,
            0.0,
        )

        envelope = np.maximum(
            envelope,
            shape,
        )

        applied_pauses += 1

    return envelope, applied_pauses


def generate_viseme_motion(
    baseline_path: Path,
    viseme_path: Path,
    output_path: Path,
) -> dict:
    if not baseline_path.is_file():
        raise FileNotFoundError(
            f"Baseline not found: {baseline_path}"
        )

    if not viseme_path.is_file():
        raise FileNotFoundError(
            f"Viseme file not found: {viseme_path}"
        )

    if baseline_path.resolve() == output_path.resolve():
        raise ValueError(
            "Output cannot overwrite baseline motion"
        )

    baseline = np.load(
        baseline_path,
        allow_pickle=False,
    )

    if (
        baseline.ndim != 2
        or baseline.shape[1] != 7
        or baseline.shape[0] == 0
    ):
        raise ValueError(
            "Baseline motion must contain seven channels"
        )

    if not np.isfinite(baseline).all():
        raise ValueError(
            "Baseline contains invalid values"
        )

    viseme_data = json.loads(
        viseme_path.read_text(encoding="utf-8")
    )

    segments = viseme_data.get("segments", [])
    pauses = viseme_data.get("pauses", [])

    if not isinstance(segments, list):
        raise ValueError(
            "Viseme segments must be a list"
        )

    if not isinstance(pauses, list):
        raise ValueError(
            "Speech pauses must be a list"
        )

    frame_count = baseline.shape[0]

    motion = baseline.astype(
        np.float64,
        copy=True,
    )

    target_open, target_width, confidence = (
        build_viseme_targets(
            segments,
            frame_count,
            FPS,
        )
    )

    pause_envelope, applied_pauses = (
        build_pause_envelope(
            pauses,
            frame_count,
            FPS,
        )
    )

    baseline_open = clamp_signal(
        motion[:, 0]
    )

    baseline_width = clamp_signal(
        motion[:, 1]
    )

    blend = confidence * VISEME_STRENGTH

    combined_open = (
        (1.0 - blend) * baseline_open
        + blend * target_open
    )

    combined_width = (
        (1.0 - blend) * baseline_width
        + blend * target_width
    )

    combined_open = (
        (1.0 - pause_envelope) * combined_open
        + pause_envelope * PAUSE_MOUTH_OPEN
    )

    combined_width = (
        (1.0 - pause_envelope) * combined_width
        + pause_envelope * PAUSE_MOUTH_WIDTH
    )

    smooth_open = smooth_transition(
        combined_open,
        ATTACK_SECONDS,
        RELEASE_SECONDS,
        FPS,
    )

    smooth_width = smooth_transition(
        combined_width,
        ATTACK_SECONDS,
        RELEASE_SECONDS,
        FPS,
    )

    smooth_open = limit_motion_speed(
        smooth_open,
        MAX_CHANGE_PER_SECOND,
        FPS,
    )

    smooth_width = limit_motion_speed(
        smooth_width,
        MAX_CHANGE_PER_SECOND,
        FPS,
    )

    motion[:, 0] = clamp_signal(smooth_open)
    motion[:, 1] = clamp_signal(smooth_width)

    output = motion.astype(np.float32)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.save(output_path, output)

    frame_changes = np.abs(
        np.diff(output[:, 0])
    )

    return {
        "output_file": str(output_path),
        "frames": int(frame_count),
        "fps": FPS,
        "dimensions": int(output.shape[1]),
        "duration": round(
            frame_count / FPS,
            3,
        ),
        "viseme_segments": len(segments),
        "detected_pauses": len(pauses),
        "applied_pauses": applied_pauses,
        "mouth_open_min": round(
            float(output[:, 0].min()),
            4,
        ),
        "mouth_open_max": round(
            float(output[:, 0].max()),
            4,
        ),
        "mean_frame_change": round(
            float(frame_changes.mean())
            if frame_changes.size
            else 0.0,
            5,
        ),
        "max_frame_change": round(
            float(frame_changes.max())
            if frame_changes.size
            else 0.0,
            5,
        ),
        "other_parameters_preserved": bool(
            np.array_equal(
                baseline[:, 2:],
                output[:, 2:],
            )
        ),
        "smoothing": {
            "attack_seconds": ATTACK_SECONDS,
            "release_seconds": RELEASE_SECONDS,
            "max_change_per_second": MAX_CHANGE_PER_SECOND,
            "viseme_strength": VISEME_STRENGTH,
            "minimum_pause_seconds": MIN_PAUSE_DURATION,
            "pause_fade_seconds": PAUSE_FADE_SECONDS,
        },
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
