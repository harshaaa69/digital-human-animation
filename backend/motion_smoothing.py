
import argparse
import json
from pathlib import Path

import numpy as np

from backend.motion_evaluation import calculate_metrics


FPS = 30
MOTION_DIMENSIONS = 7


def gaussian_kernel(
    window_size: int,
    sigma: float,
) -> np.ndarray:
    if window_size < 3 or window_size % 2 == 0:
        raise ValueError(
            "Window size must be an odd number of at least 3"
        )

    if sigma <= 0:
        raise ValueError("Sigma must be positive")

    positions = np.arange(
        -(window_size // 2),
        window_size // 2 + 1,
        dtype=np.float64,
    )

    kernel = np.exp(
        -(positions ** 2) / (2 * sigma ** 2)
    )

    return kernel / kernel.sum()


def smooth_channel(
    values: np.ndarray,
    kernel: np.ndarray,
) -> np.ndarray:
    padding = len(kernel) // 2

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


def generate_smoothed_motion(
    input_path: Path,
    output_path: Path,
    window_size: int = 5,
    sigma: float = 1.2,
    strength: float = 0.7,
) -> dict:
    if not 0 <= strength <= 1:
        raise ValueError(
            "Smoothing strength must be between 0 and 1"
        )

    if not input_path.is_file():
        raise FileNotFoundError(
            f"Motion file not found: {input_path}"
        )

    motion = np.load(
        input_path,
        allow_pickle=False,
    )

    if (
        motion.ndim != 2
        or motion.shape[1] != MOTION_DIMENSIONS
        or motion.shape[0] == 0
    ):
        raise ValueError(
            "Motion must contain frames with 7 parameters"
        )

    if not np.isfinite(motion).all():
        raise ValueError(
            "Motion contains invalid values"
        )

    kernel = gaussian_kernel(
        window_size,
        sigma,
    )

    original = motion.astype(
        np.float64,
        copy=True,
    )

    smoothed = original.copy()

    for column in [0, 1]:
        filtered = smooth_channel(
            original[:, column],
            kernel,
        )

        smoothed[:, column] = (
            (1 - strength) * original[:, column]
            + strength * filtered
        )

        smoothed[:, column] = np.clip(
            smoothed[:, column],
            0,
            1,
        )

    smoothed = smoothed.astype(np.float32)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.save(output_path, smoothed)

    original_metrics = calculate_metrics(
        original,
        FPS,
    )

    smoothed_metrics = calculate_metrics(
        smoothed.astype(np.float64),
        FPS,
    )

    original_change = original_metrics[
        "mean_frame_change"
    ]

    smoothed_change = smoothed_metrics[
        "mean_frame_change"
    ]

    reduction = (
        (original_change - smoothed_change)
        / original_change
        * 100
        if original_change > 0
        else 0
    )

    result = {
        "input_file": input_path.name,
        "output_file": output_path.name,
        "frames": int(smoothed.shape[0]),
        "fps": FPS,
        "duration": round(
            smoothed.shape[0] / FPS,
            2,
        ),
        "dimensions": int(smoothed.shape[1]),
        "smoothing": {
            "method": "gaussian",
            "window_size": window_size,
            "sigma": sigma,
            "strength": strength,
            "parameters": [
                "mouth_open",
                "mouth_width",
            ],
        },
        "original": original_metrics,
        "smoothed": smoothed_metrics,
        "comparison": {
            "mean_mouth_difference": round(
                float(
                    np.mean(
                        np.abs(
                            original[:, 0]
                            - smoothed[:, 0]
                        )
                    )
                ),
                5,
            ),
            "frame_change_reduction_percent": round(
                reduction,
                2,
            ),
            "other_parameters_preserved": bool(
                np.array_equal(
                    original[:, 2:],
                    smoothed[:, 2:],
                )
            ),
        },
    }

    return result


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--report",
        type=Path,
    )

    parser.add_argument(
        "--window",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--sigma",
        type=float,
        default=1.2,
    )

    parser.add_argument(
        "--strength",
        type=float,
        default=0.7,
    )

    args = parser.parse_args()

    result = generate_smoothed_motion(
        args.input,
        args.output,
        args.window,
        args.sigma,
        args.strength,
    )

    if args.report:
        args.report.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        args.report.write_text(
            json.dumps(result, indent=2),
            encoding="utf-8",
        )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
