
import argparse
import json
from pathlib import Path

import numpy as np


PARAMETERS = [
    "mouth_open",
    "mouth_width",
    "brow_raise",
    "head_yaw",
    "head_pitch",
    "head_roll",
    "gesture_intensity",
]


def load_motion(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Motion file not found: {path}")

    motion = np.load(path, allow_pickle=False)

    if motion.ndim != 2 or motion.shape[1] != len(PARAMETERS):
        raise ValueError("Motion file must contain 7 parameters")

    if not np.isfinite(motion).all():
        raise ValueError("Motion contains invalid values")

    return motion.astype(np.float64)


def calculate_metrics(motion: np.ndarray, fps: int) -> dict:
    mouth = motion[:, 0]
    mouth_width = motion[:, 1]

    frame_changes = np.diff(mouth)
    time_velocity = frame_changes * fps

    acceleration = np.diff(time_velocity) * fps

    return {
        "frames": int(motion.shape[0]),
        "fps": fps,
        "duration": round(motion.shape[0] / fps, 3),
        "mean_mouth_opening": round(float(np.mean(mouth)), 5),
        "max_mouth_opening": round(float(np.max(mouth)), 5),
        "min_mouth_opening": round(float(np.min(mouth)), 5),
        "mouth_opening_std": round(float(np.std(mouth)), 5),
        "mean_mouth_width": round(float(np.mean(mouth_width)), 5),
        "mean_frame_change": round(
            float(np.mean(np.abs(frame_changes)))
            if frame_changes.size
            else 0.0,
            5,
        ),
        "maximum_frame_change": round(
            float(np.max(np.abs(frame_changes)))
            if frame_changes.size
            else 0.0,
            5,
        ),
        "mean_velocity": round(
            float(np.mean(np.abs(time_velocity)))
            if time_velocity.size
            else 0.0,
            5,
        ),
        "mean_acceleration": round(
            float(np.mean(np.abs(acceleration)))
            if acceleration.size
            else 0.0,
            5,
        ),
    }


def evaluate_motion(
    baseline_path: Path,
    enhanced_path: Path,
    output_path: Path,
    fps: int = 30,
) -> dict:
    if fps <= 0:
        raise ValueError("FPS must be positive")

    baseline = load_motion(baseline_path)
    enhanced = load_motion(enhanced_path)

    if baseline.shape != enhanced.shape:
        raise ValueError(
            "Motion files must have identical dimensions"
        )

    baseline_metrics = calculate_metrics(baseline, fps)
    enhanced_metrics = calculate_metrics(enhanced, fps)

    absolute_difference = np.abs(baseline - enhanced)

    parameter_differences = {}

    for index, parameter in enumerate(PARAMETERS):
        parameter_differences[parameter] = round(
            float(np.mean(absolute_difference[:, index])),
            5,
        )

    correlation = None

    if (
        np.std(baseline[:, 0]) > 1e-10
        and np.std(enhanced[:, 0]) > 1e-10
    ):
        correlation = round(
            float(
                np.corrcoef(
                    baseline[:, 0],
                    enhanced[:, 0],
                )[0, 1]
            ),
            5,
        )

    result = {
        "evaluation": "Baseline vs Viseme-Enhanced Motion",
        "baseline_file": baseline_path.name,
        "enhanced_file": enhanced_path.name,
        "baseline": baseline_metrics,
        "viseme_enhanced": enhanced_metrics,
        "comparison": {
            "mean_absolute_mouth_difference": round(
                float(np.mean(absolute_difference[:, 0])),
                5,
            ),
            "maximum_mouth_difference": round(
                float(np.max(absolute_difference[:, 0])),
                5,
            ),
            "mouth_motion_correlation": correlation,
            "parameter_mae": parameter_differences,
            "other_parameters_preserved": bool(
                np.array_equal(
                    baseline[:, 2:],
                    enhanced[:, 2:],
                )
            ),
        },
        "interpretation": (
            "These metrics measure movement characteristics "
            "and differences between methods. They do not "
            "establish accurate lip synchronization or "
            "realistic facial animation."
        ),
    }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    return result


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--baseline",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--enhanced",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=30,
    )

    args = parser.parse_args()

    result = evaluate_motion(
        args.baseline,
        args.enhanced,
        args.output,
        args.fps,
    )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
