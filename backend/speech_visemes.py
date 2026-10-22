
import argparse
import json
import re
from pathlib import Path
from typing import Any, cast

import cmudict
import librosa
import numpy as np
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

VOWEL_PHONEMES = {
    "AA", "AE", "AH", "AO", "AW", "AY",
    "EH", "ER", "EY", "IH", "IY",
    "OW", "OY", "UH", "UW",
}

STOP_PHONEMES = {
    "P", "B", "T", "D", "K", "G",
}

NASAL_PHONEMES = {
    "M", "N", "NG",
}

FRICATIVE_PHONEMES = {
    "F", "V", "TH", "DH", "S",
    "Z", "SH", "ZH", "HH",
}


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def normalize_phoneme(phoneme: str) -> str:
    return re.sub(r"\d", "", phoneme.upper())


def classify_phoneme(phoneme: str) -> str:
    phoneme = normalize_phoneme(phoneme)

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


def get_phoneme_weight(phoneme: str) -> float:
    phoneme = normalize_phoneme(phoneme)

    if phoneme in VOWEL_PHONEMES:
        return 1.65

    if phoneme in STOP_PHONEMES:
        return 0.65

    if phoneme in NASAL_PHONEMES:
        return 1.05

    if phoneme in FRICATIVE_PHONEMES:
        return 1.15

    if phoneme in {"L", "R", "W", "Y"}:
        return 1.2

    return 1.0


def get_letter_weight(letter: str) -> float:
    if letter.upper() in "AEIOUY":
        return 1.4

    return 1.0


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
            token = tokenizer.convert_ids_to_tokens(
                int(token_id)
            )

            if isinstance(token, str):
                if token == delimiter:
                    token = "|"

                if token == "|" or (
                    len(token) == 1
                    and token.isalpha()
                ):
                    character_segments.append(
                        {
                            "token": token,
                            "start": start_index
                            * frame_duration,
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
            item["token"]
            for item in characters
        )

        start = float(characters[0]["start"])
        end = float(characters[-1]["end"])

        if word and end > start:
            words.append(
                {
                    "word": word,
                    "start": start,
                    "end": end,
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


def allocate_phoneme_intervals(
    units: list[str],
    start: float,
    end: float,
    source: str,
) -> list[tuple[str, float, float]]:
    if not units or end <= start:
        return []

    if source == "cmudict":
        weights = np.asarray(
            [
                get_phoneme_weight(unit)
                for unit in units
            ],
            dtype=np.float64,
        )
    else:
        weights = np.asarray(
            [
                get_letter_weight(unit)
                for unit in units
            ],
            dtype=np.float64,
        )

    total_weight = float(weights.sum())

    if total_weight <= 0:
        return []

    durations = (
        weights / total_weight
    ) * (end - start)

    boundaries = np.concatenate(
        (
            np.asarray([start]),
            start + np.cumsum(durations),
        )
    )

    boundaries[-1] = end

    intervals = []

    for index, unit in enumerate(units):
        interval_start = float(
            boundaries[index]
        )

        interval_end = float(
            boundaries[index + 1]
        )

        if interval_end > interval_start:
            intervals.append(
                (
                    unit,
                    interval_start,
                    interval_end,
                )
            )

    return intervals


def convert_words_to_visemes(
    words: list[dict],
) -> tuple[list[dict], int]:
    segments = []
    fallback_words = 0

    for entry in words:
        word = str(entry["word"]).lower()
        start = float(entry["start"])
        end = float(entry["end"])

        if (
            not np.isfinite(start)
            or not np.isfinite(end)
            or end <= start
        ):
            continue

        pronunciations = PRONUNCIATIONS.get(word)

        if pronunciations:
            units = [
                normalize_phoneme(phoneme)
                for phoneme in pronunciations[0]
            ]

            source = "cmudict"
        else:
            units = list(word.upper())
            source = "letter_fallback"
            fallback_words += 1

        intervals = allocate_phoneme_intervals(
            units,
            start,
            end,
            source,
        )

        for unit, unit_start, unit_end in intervals:
            viseme = (
                classify_phoneme(unit)
                if source == "cmudict"
                else classify_letter(unit)
            )

            segments.append(
                {
                    "word": entry["word"],
                    "phoneme": (
                        unit
                        if source == "cmudict"
                        else None
                    ),
                    "token": unit,
                    "viseme": viseme,
                    "start": round(unit_start, 4),
                    "end": round(unit_end, 4),
                    "source": source,
                }
            )

    segments.sort(
        key=lambda segment: (
            segment["start"],
            segment["end"],
        )
    )

    return segments, fallback_words


def get_speech_pauses(
    words: list[dict],
    duration: float,
    minimum_pause: float = 0.08,
) -> list[dict]:
    pauses = []
    previous_end = 0.0

    for word in words:
        start = float(word["start"])
        end = float(word["end"])

        if start - previous_end >= minimum_pause:
            pauses.append(
                {
                    "start": round(
                        previous_end,
                        4,
                    ),
                    "end": round(start, 4),
                }
            )

        previous_end = max(
            previous_end,
            end,
        )

    if duration - previous_end >= minimum_pause:
        pauses.append(
            {
                "start": round(
                    previous_end,
                    4,
                ),
                "end": round(
                    duration,
                    4,
                ),
            }
        )

    return pauses


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
        Wav2Vec2Processor.from_pretrained(
            MODEL_NAME
        ),
    )

    model = cast(
        Any,
        Wav2Vec2ForCTC.from_pretrained(
            MODEL_NAME
        ),
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

    pauses = get_speech_pauses(
        words,
        duration,
    )

    phoneme_segments = sum(
        segment["source"] == "cmudict"
        for segment in segments
    )

    result = {
        "audio_file": audio_path.name,
        "duration": round(duration, 4),
        "sample_rate": sample_rate,
        "model": MODEL_NAME,
        "device": str(device),
        "transcription": transcription,
        "alignment_type": "ctc_word_approximate",
        "viseme_type": "duration_weighted_phoneme_approximation",
        "phoneme_timing": "weighted_within_ctc_word",
        "words": words,
        "word_count": len(words),
        "fallback_word_count": fallback_words,
        "phoneme_segments": phoneme_segments,
        "pauses": pauses,
        "pause_count": len(pauses),
        "segments": segments,
    }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            result,
            indent=2,
        ),
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
                "phoneme_segments": result[
                    "phoneme_segments"
                ],
                "fallback_words": result[
                    "fallback_word_count"
                ],
                "pause_count": result["pause_count"],
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
