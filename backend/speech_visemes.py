
import argparse
import json
import re
from pathlib import Path
from typing import Any, cast

import cmudict
import librosa
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor


MODEL_NAME = "facebook/wav2vec2-base-960h"
SAMPLE_RATE = 16000
MAX_DURATION = 20

PRONUNCIATIONS = cmudict.dict()

PHONEME_GROUPS = {
    "closed_lips": {"P", "B", "M"},
    "teeth_lip": {"F", "V"},
    "rounded": {"OW", "UW", "UH", "AO", "W"},
    "wide": {"IY", "IH", "EY", "EH", "Y"},
    "open": {"AA", "AE", "AH", "AW", "AY", "ER"},
    "tongue": {"L", "T", "D", "N", "TH", "DH"},
    "consonant": {
        "CH", "JH", "SH", "ZH", "S",
        "Z", "K", "G", "NG", "HH", "R",
    },
}

LETTER_GROUPS = {
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


def classify_phoneme(phoneme: str) -> str:
    phoneme = re.sub(r"\d", "", phoneme.upper())

    for viseme, phonemes in PHONEME_GROUPS.items():
        if phoneme in phonemes:
            return viseme

    return "consonant"


def classify_letter(letter: str) -> str:
    letter = letter.upper()

    for viseme, letters in LETTER_GROUPS.items():
        if letter in letters:
            return viseme

    return "rest"


def extract_ctc_words(
    token_ids: list[int],
    processor: Any,
    duration: float,
) -> list[dict]:
    if not token_ids:
        return []

    tokenizer = processor.tokenizer
    blank_id = tokenizer.pad_token_id
    delimiter = tokenizer.word_delimiter_token or "|"

    frame_duration = duration / len(token_ids)
    character_segments = []

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

                if token == "|" or (
                    len(token) == 1 and token.isalpha()
                ):
                    character_segments.append(
                        {
                            "token": token,
                            "start": start_index * frame_duration,
                            "end": min(
                                duration,
                                end_index * frame_duration,
                            ),
                        }
                    )

        index += 1

    words = []
    characters = []

    def finish_word() -> None:
        if not characters:
            return

        word = "".join(
            item["token"] for item in characters
        )

        words.append(
            {
                "word": word,
                "start": characters[0]["start"],
                "end": characters[-1]["end"],
            }
        )

        characters.clear()

    for item in character_segments:
        if item["token"] == "|":
            finish_word()
        else:
            characters.append(item)

    finish_word()

    return words


def convert_words_to_visemes(
    words: list[dict],
) -> tuple[list[dict], int]:
    segments = []
    fallback_words = 0

    for entry in words:
        word = entry["word"].lower()
        start = float(entry["start"])
        end = float(entry["end"])

        if end <= start:
            continue

        pronunciations = PRONUNCIATIONS.get(word)

        if pronunciations:
            units = [
                re.sub(r"\d", "", phoneme)
                for phoneme in pronunciations[0]
            ]
            source = "cmudict"
        else:
            units = list(word.upper())
            source = "letter_fallback"
            fallback_words += 1

        if not units:
            continue

        step = (end - start) / len(units)

        for index, unit in enumerate(units):
            unit_start = start + index * step
            unit_end = start + (index + 1) * step

            viseme = (
                classify_phoneme(unit)
                if source == "cmudict"
                else classify_letter(unit)
            )

            segments.append(
                {
                    "word": entry["word"],
                    "phoneme": (
                        unit if source == "cmudict"
                        else None
                    ),
                    "token": unit,
                    "viseme": viseme,
                    "start": round(unit_start, 3),
                    "end": round(unit_end, 3),
                    "source": source,
                }
            )

    return segments, fallback_words


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

    processor = cast(
        Any,
        Wav2Vec2Processor.from_pretrained(MODEL_NAME),
    )

    model = cast(
        Any,
        Wav2Vec2ForCTC.from_pretrained(MODEL_NAME),
    )

    model = model.to(device)
    model.eval()

    inputs = processor(
        audio,
        sampling_rate=sample_rate,
        return_tensors="pt",
    )

    with torch.inference_mode():
        logits = model(
            inputs.input_values.to(device)
        ).logits

    predicted_ids = (
        torch.argmax(logits, dim=-1)
        .squeeze(0)
        .cpu()
        .tolist()
    )

    transcription = processor.batch_decode(
        [predicted_ids]
    )[0]

    words = extract_ctc_words(
        predicted_ids,
        processor,
        duration,
    )

    segments, fallback_words = (
        convert_words_to_visemes(words)
    )

    phoneme_segments = sum(
        segment["source"] == "cmudict"
        for segment in segments
    )

    result = {
        "audio_file": audio_path.name,
        "duration": round(duration, 3),
        "sample_rate": sample_rate,
        "model": MODEL_NAME,
        "device": str(device),
        "transcription": transcription,
        "alignment_type": "ctc_word_approximate",
        "viseme_type": "phoneme_aware_approximation",
        "phoneme_timing": "uniform_within_ctc_word",
        "words": words,
        "word_count": len(words),
        "fallback_word_count": fallback_words,
        "phoneme_segments": phoneme_segments,
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
                "word_count": result["word_count"],
                "phoneme_segments": result["phoneme_segments"],
                "fallback_words": result[
                    "fallback_word_count"
                ],
                "total_segments": len(
                    result["segments"]
                ),
                "output_file": str(output_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
