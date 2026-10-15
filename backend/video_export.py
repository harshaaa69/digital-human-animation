
from pathlib import Path
import shutil
import subprocess


def get_ffmpeg_path() -> str:
    ffmpeg_path = shutil.which("ffmpeg")

    if not ffmpeg_path:
        raise RuntimeError(
            "FFmpeg is not installed or is not available in PATH"
        )

    return ffmpeg_path


def export_mp4(
    recording_path: Path,
    audio_path: Path,
    output_path: Path,
) -> dict:
    if not recording_path.is_file():
        raise FileNotFoundError("Recorded video not found")

    if not audio_path.is_file():
        raise FileNotFoundError("Audio file not found")

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    command = [
        get_ffmpeg_path(),
        "-y",
        "-i",
        str(recording_path),
        "-i",
        str(audio_path),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-r",
        "30",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-shortest",
        "-movflags",
        "+faststart",
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

        error_message = result.stderr.strip()

        raise RuntimeError(
            error_message[-1500:]
            if error_message
            else "FFmpeg video encoding failed"
        )

    if not output_path.exists():
        raise RuntimeError("Video output was not created")

    return {
        "video_name": output_path.name,
        "size_bytes": output_path.stat().st_size,
        "format": "MP4",
        "video_codec": "H.264",
        "audio_codec": "AAC",
        "fps": 30,
    }
