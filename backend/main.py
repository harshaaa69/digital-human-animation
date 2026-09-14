from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from backend.audio_utils import (
    convert_to_standard_wav,
    get_audio_metadata,
)
from backend.speech_features import extract_speech_features

app = FastAPI(
    title="Digital Human Animation API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent.parent

UPLOAD_DIR = BASE_DIR / "data" / "uploads"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
FEATURE_DIR = BASE_DIR / "data" / "features"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
FEATURE_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_AUDIO_TYPES = {
    "audio/mpeg",
    "audio/wav",
    "audio/x-wav",
    "audio/mp4",
    "audio/x-m4a",
    "audio/aac",
}

MAX_FILE_SIZE = 25 * 1024 * 1024


@app.get("/")
def root():
    return {
        "message": "Digital Human Animation API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "ok"
    }


@app.post("/upload-audio")
async def upload_audio(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_AUDIO_TYPES:
        raise HTTPException(
            status_code=400,
            detail="Unsupported audio format"
        )

    content = await file.read()

    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail="Audio file must be smaller than 25 MB"
        )

    original_name = Path(file.filename or "audio").name
    extension = Path(original_name).suffix.lower()

    unique_id = uuid4().hex

    stored_name = f"{unique_id}{extension}"
    uploaded_path = UPLOAD_DIR / stored_name

    uploaded_path.write_bytes(content)

    processed_name = f"{unique_id}.wav"
    processed_path = PROCESSED_DIR / processed_name

    try:
        convert_to_standard_wav(
            uploaded_path,
            processed_path
        )

        metadata = get_audio_metadata(
            processed_path
        )

    except Exception:
        if uploaded_path.exists():
            uploaded_path.unlink()

        if processed_path.exists():
            processed_path.unlink()

        raise HTTPException(
            status_code=500,
            detail="Audio preprocessing failed"
        )

    return {
        "message": "Audio uploaded and processed successfully",
        "original_name": original_name,
        "stored_name": stored_name,
        "processed_name": processed_name,
        "size": len(content),
        "content_type": file.content_type,
        "metadata": metadata,
    }


@app.post("/extract-features/{processed_name}")
def extract_features(processed_name: str):
    safe_name = Path(processed_name).name

    if safe_name != processed_name:
        raise HTTPException(
            status_code=400,
            detail="Invalid processed filename"
        )

    if not safe_name.lower().endswith(".wav"):
        raise HTTPException(
            status_code=400,
            detail="Processed file must be a WAV file"
        )

    processed_path = PROCESSED_DIR / safe_name

    if not processed_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Processed audio file not found"
        )

    feature_name = f"{processed_path.stem}.npy"
    feature_path = FEATURE_DIR / feature_name

    try:
        result = extract_speech_features(
            processed_path,
            feature_path,
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Speech feature extraction failed: {error}"
        )

    return {
        "message": "Speech features extracted successfully",
        "processed_name": safe_name,
        **result,
    }