from pathlib import Path
from typing import Any

import librosa
import numpy as np
import torch
from transformers import Wav2Vec2Model, Wav2Vec2Processor


MODEL_NAME = "facebook/wav2vec2-base-960h"

processor: Any = None
model: Any = None
device: torch.device | None = None


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def load_wav2vec() -> tuple[Any, Any, torch.device]:
    global processor, model, device

    if processor is None or model is None or device is None:
        device = get_device()

        processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)

        model = Wav2Vec2Model.from_pretrained(MODEL_NAME)
        model = model.to(device)
        model.eval()

    return processor, model, device


def extract_speech_features(
    audio_path: Path,
    output_path: Path,
) -> dict:
    audio, sample_rate = librosa.load(
        audio_path,
        sr=16000,
        mono=True,
    )

    wav2vec_processor, wav2vec_model, active_device = load_wav2vec()

    inputs = wav2vec_processor(
        audio,
        sampling_rate=sample_rate,
        return_tensors="pt",
        padding=True,
    )

    input_values = inputs.input_values.to(active_device)

    with torch.no_grad():
        outputs = wav2vec_model(input_values)

    features = outputs.last_hidden_state.squeeze(0).cpu().numpy()
    features = features.astype(np.float32)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.save(
        output_path,
        features,
    )

    return {
        "frames": int(features.shape[0]),
        "feature_dimension": int(features.shape[1]),
        "feature_file": output_path.name,
        "device": str(active_device),
    }