from pathlib import Path

import librosa
import numpy as np


MOTION_FPS = 30


def normalize(values: np.ndarray) -> np.ndarray:
    minimum = np.min(values)
    maximum = np.max(values)

    if maximum - minimum < 1e-8:
        return np.zeros_like(values)

    return (values - minimum) / (maximum - minimum)


def smooth(values: np.ndarray, window_size: int = 7) -> np.ndarray:
    if len(values) < window_size:
        return values

    kernel = np.ones(window_size) / window_size
    return np.convolve(values, kernel, mode="same")


def resize_signal(values: np.ndarray, target_frames: int) -> np.ndarray:
    if len(values) == target_frames:
        return values

    source_positions = np.linspace(
        0,
        1,
        num=len(values),
    )

    target_positions = np.linspace(
        0,
        1,
        num=target_frames,
    )

    return np.interp(
        target_positions,
        source_positions,
        values,
    )


def generate_motion(
    audio_path: Path,
    output_path: Path,
) -> dict:
    audio, sample_rate = librosa.load(
        audio_path,
        sr=16000,
        mono=True,
    )

    duration = librosa.get_duration(
        y=audio,
        sr=sample_rate,
    )

    target_frames = max(
        1,
        int(round(duration * MOTION_FPS)),
    )

    rms = librosa.feature.rms(
        y=audio,
        frame_length=1024,
        hop_length=512,
    )[0]

    spectral_centroid = librosa.feature.spectral_centroid(
        y=audio,
        sr=sample_rate,
        n_fft=1024,
        hop_length=512,
    )[0]

    zero_crossing = librosa.feature.zero_crossing_rate(
        audio,
        frame_length=1024,
        hop_length=512,
    )[0]

    onset = librosa.onset.onset_strength(
        y=audio,
        sr=sample_rate,
        hop_length=512,
    )

    rms = smooth(normalize(rms))
    spectral_centroid = smooth(normalize(spectral_centroid))
    zero_crossing = smooth(normalize(zero_crossing))
    onset = smooth(normalize(onset))

    rms = resize_signal(rms, target_frames)
    spectral_centroid = resize_signal(
        spectral_centroid,
        target_frames,
    )
    zero_crossing = resize_signal(
        zero_crossing,
        target_frames,
    )
    onset = resize_signal(
        onset,
        target_frames,
    )

    time = np.arange(target_frames) / MOTION_FPS

    mouth_open = np.clip(
        rms * 0.85 + onset * 0.15,
        0,
        1,
    )

    mouth_width = np.clip(
        spectral_centroid * 0.6 + rms * 0.4,
        0,
        1,
    )

    brow_raise = np.clip(
        onset * 0.65 + spectral_centroid * 0.35,
        0,
        1,
    )

    head_yaw = (
        np.sin(time * 0.75) *
        (0.15 + onset * 0.2)
    )

    head_pitch = (
        np.sin(time * 0.48 + 1.2) *
        (0.08 + rms * 0.12)
    )

    head_roll = (
        np.sin(time * 0.31 + 0.5) *
        (0.04 + zero_crossing * 0.06)
    )

    gesture_intensity = np.clip(
        onset * 0.55 +
        rms * 0.30 +
        spectral_centroid * 0.15,
        0,
        1,
    )

    motion = np.column_stack(
        [
            mouth_open,
            mouth_width,
            brow_raise,
            head_yaw,
            head_pitch,
            head_roll,
            gesture_intensity,
        ]
    ).astype(np.float32)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.save(
        output_path,
        motion,
    )

    return {
        "motion_file": output_path.name,
        "frames": int(motion.shape[0]),
        "fps": MOTION_FPS,
        "duration": round(duration, 2),
        "motion_dimensions": int(motion.shape[1]),
        "parameters": [
            "mouth_open",
            "mouth_width",
            "brow_raise",
            "head_yaw",
            "head_pitch",
            "head_roll",
            "gesture_intensity",
        ],
    }