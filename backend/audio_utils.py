from pathlib import Path

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