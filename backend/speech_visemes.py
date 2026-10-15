
import argparse
import json
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor


MODEL_NAME = "facebook/wav2vec2-base-960h"
SAMPLE_RATE = 16000
MAX_DURATION = 20

VISEME_GROUPS = {
    "closed_lips": set("BMP"),
    "teeth_lip": set("FV"),
    "rounded": set("OUWQ"),
    "wide": set("EIY"),
    "open": set("A"),
    "tongue": set("LTDN"),
    "consonant": set("CGHJKRSXZ"),
}


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def classify_token(token: str) -> str:
    token = token.upper()

    if token == "|":
        return "rest"

    for viseme, characters in VISEME_GROUPS.items():
        if token in characters:
            return viseme

    return "rest"


def get_token_segments(
    token_ids: list[int],
    processor: Any,
    duration: float,
) -> list[dict]:
    if not token_ids:
        return []

    tokenizer = processor.tokenizer
    blank_id = tokenizer.pad_token_id
    delimiter = tokenizer.word_delimiter_token or "|"

    seconds_per_frame = duration / len(token_ids)
    segments = []

    index = 0

    while index < len(token_ids):
        token_id = token_ids[index]
        start_index = index

        while (
            index + 1 < len(token_ids)
            and token_ids[index + 1] == token_id
        ):
            index += 1

        end_index = index + 1

        if token_id != blank_id:
            token = tokenizer.convert_ids_to_tokens(token_id)

            if isinstance(token, str):
                if token == delimiter:
                    token = "|"

                if len(token) == 1:
                    segments.append(
                        {
                            "token": token,
                            "viseme": classify_token(token),
                            "start": round(
                                start_index * seconds_per_frame,
                                3,
                            ),
                            "end": round(
                                min(
                                    duration,
                                    end_index * seconds_per_frame,
                                ),
                                3,
                            ),
                        }
                    )

        index += 1

    return segments


def extract_visemes(
    audio_path: Path,
    output_path: Path,
) -> dict:
    if not audio_path.is_file():
        raise FileNotFoundError(
            f"Audio file not found: {audio_path}"
        )

    audio, sample_rate = librosa.load(
        audio_path,
        sr=SAMPLE_RATE,
        mono=True,
    )

    duration = len(audio) / sample_rate

    if duration <= 0:
        raise ValueError("Audio file is empty")

    if duration > MAX_DURATION:
        raise ValueError(
            f"Audio must not exceed {MAX_DURATION} seconds"
        )

    device = get_device()

    processor = Wav2Vec2Processor.from_pretrained(
        MODEL_NAME
    )

    model = Wav2Vec2ForCTC.from_pretrained(
        MODEL_NAME
    )

    model = model.to(device)
    model.eval()

    inputs = processor(
        audio,
        sampling_rate=sample_rate,
        return_tensors="pt",
    )

    input_values = inputs.input_values.to(device)

    with torch.inference_mode():
        outputs = model(input_values)

    logits = outputs.logits

    predicted_ids = (
        torch.argmax(logits, dim=-1)
        .squeeze(0)
        .cpu()
        .tolist()
    )

    transcription = processor.batch_decode(
        np.asarray([predicted_ids])
    )[0]

    segments = get_token_segments(
        predicted_ids,
        processor,
        duration,
    )

    result = {
        "audio_file": audio_path.name,
        "duration": round(duration, 3),
        "sample_rate": sample_rate,
        "model": MODEL_NAME,
        "device": str(device),
        "transcription": transcription,
        "alignment_type": "ctc_greedy_approximate",
        "viseme_type": "character_based_approximation",
        "segments": segments,
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
        "--audio",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output",
        type=Path,
    )

    args = parser.parse_args()

    audio_path = args.audio.resolve()

    output_path = (
        args.output.resolve()
        if args.output
        else Path("data/visemes")
        / f"{audio_path.stem}_visemes.json"
    )

    result = extract_visemes(
        audio_path,
        output_path,
    )

    print(
        json.dumps(
            {
                "audio_file": result["audio_file"],
                "duration": result["duration"],
                "device": result["device"],
                "transcription": result["transcription"],
                "segments": len(result["segments"]),
                "output_file": str(output_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
