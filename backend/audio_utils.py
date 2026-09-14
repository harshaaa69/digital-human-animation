from pathlib import Path

import librosa
import soundfile as sf
from pydub import AudioSegment


def convert_to_standard_wav(input_path: Path, output_path: Path) -> Path:
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_channels(1)
    audio = audio.set_frame_rate(16000)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    audio.export(
        output_path,
        format="wav"
    )

    return output_path


def get_audio_metadata(audio_path: Path) -> dict:
    data, sample_rate = librosa.load(
        audio_path,
        sr=None,
        mono=False
    )

    info = sf.info(audio_path)

    duration = librosa.get_duration(
        y=data,
        sr=sample_rate
    )

    return {
        "duration": round(duration, 2),
        "sample_rate": sample_rate,
        "channels": info.channels,
        "frames": info.frames,
        "format": info.format,
        "subtype": info.subtype,
    }