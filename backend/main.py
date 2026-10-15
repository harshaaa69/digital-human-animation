
from pathlib import Path
from uuid import uuid4

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from backend.audio_utils import (
    convert_to_standard_wav,
    create_test_segment,
    get_audio_metadata,
)
from backend.motion_generator import generate_motion
from backend.speech_features import extract_speech_features
from backend.video_export import export_mp4

app = FastAPI(
    title="Digital Human Animation API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent.parent

UPLOAD_DIR = BASE_DIR / "data" / "uploads"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
FEATURE_DIR = BASE_DIR / "data" / "features"
TEST_SEGMENT_DIR = BASE_DIR / "data" / "test_segments"
MOTION_DIR = BASE_DIR / "data" / "motion"
OUTPUT_DIR = BASE_DIR / "outputs"
TEMP_VIDEO_DIR = OUTPUT_DIR / "temporary"

for directory in [
    UPLOAD_DIR,
    PROCESSED_DIR,
    FEATURE_DIR,
    TEST_SEGMENT_DIR,
    MOTION_DIR,
    OUTPUT_DIR,
    TEMP_VIDEO_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)

ALLOWED_AUDIO_TYPES = {
    "audio/mpeg",
    "audio/wav",
    "audio/x-wav",
    "audio/mp4",
    "audio/x-m4a",
    "audio/aac",
}

MAX_FILE_SIZE = 25 * 1024 * 1024
MAX_RECORDING_SIZE = 150 * 1024 * 1024

MOTION_PARAMETERS = [
    "mouth_open",
    "mouth_width",
    "brow_raise",
    "head_yaw",
    "head_pitch",
    "head_roll",
    "gesture_intensity",
]


def validate_filename(filename: str, suffix: str) -> str:
    if (
        not filename
        or filename != Path(filename).name
        or "/" in filename
        or "\\" in filename
        or not filename.lower().endswith(suffix)
    ):
        raise HTTPException(
            status_code=400,
            detail="Invalid filename",
        )

    return filename


@app.get("/")
def root():
    return {
        "message": "Digital Human Animation API is running"
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/upload-audio")
async def upload_audio(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_AUDIO_TYPES:
        raise HTTPException(
            status_code=400,
            detail="Unsupported audio format",
        )

    content = await file.read(MAX_FILE_SIZE + 1)

    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail="Audio file must be smaller than 25 MB",
        )

    original_name = Path(file.filename or "audio").name
    extension = Path(original_name).suffix.lower()

    if extension not in {
        ".mp3",
        ".wav",
        ".m4a",
        ".aac",
    }:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file extension",
        )

    unique_id = uuid4().hex

    stored_name = f"{unique_id}{extension}"
    uploaded_path = UPLOAD_DIR / stored_name

    processed_name = f"{unique_id}.wav"
    processed_path = PROCESSED_DIR / processed_name

    uploaded_path.write_bytes(content)

    try:
        convert_to_standard_wav(
            uploaded_path,
            processed_path,
        )

        metadata = get_audio_metadata(processed_path)

    except Exception:
        uploaded_path.unlink(missing_ok=True)
        processed_path.unlink(missing_ok=True)

        raise HTTPException(
            status_code=500,
            detail="Audio preprocessing failed",
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


@app.post("/create-test-segment/{processed_name}")
def create_short_test_segment(processed_name: str):
    safe_name = validate_filename(processed_name, ".wav")
    processed_path = PROCESSED_DIR / safe_name

    if not processed_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Processed audio file not found",
        )

    segment_name = f"{processed_path.stem}_15s.wav"
    segment_path = TEST_SEGMENT_DIR / segment_name

    try:
        create_test_segment(
            processed_path,
            segment_path,
            duration_seconds=15,
        )

        metadata = get_audio_metadata(segment_path)

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Test segment creation failed: {error}",
        )

    return {
        "message": "Test segment created successfully",
        "segment_name": segment_name,
        "metadata": metadata,
    }


@app.post("/extract-features/{processed_name}")
def extract_features(processed_name: str):
    safe_name = validate_filename(processed_name, ".wav")
    processed_path = PROCESSED_DIR / safe_name

    if not processed_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Processed audio file not found",
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
            detail=f"Speech feature extraction failed: {error}",
        )

    return {
        "message": "Speech features extracted successfully",
        "processed_name": safe_name,
        **result,
    }


@app.post("/generate-motion/{segment_name}")
def generate_motion_data(segment_name: str):
    safe_name = validate_filename(segment_name, ".wav")
    segment_path = TEST_SEGMENT_DIR / safe_name

    if not segment_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Test segment not found",
        )

    motion_name = f"{segment_path.stem}_motion.npy"
    motion_path = MOTION_DIR / motion_name

    try:
        result = generate_motion(
            segment_path,
            motion_path,
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Motion generation failed: {error}",
        )

    return {
        "message": "Motion generated successfully",
        "segment_name": safe_name,
        **result,
    }


@app.get("/motion-data/{motion_name}")
def get_motion_data(motion_name: str):
    safe_name = validate_filename(motion_name, ".npy")
    motion_path = MOTION_DIR / safe_name

    if not motion_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Motion file not found",
        )

    try:
        motion = np.load(
            motion_path,
            allow_pickle=False,
        )

        if motion.ndim != 2 or motion.shape[1] != 7:
            raise ValueError("Unexpected motion data shape")

        if not np.isfinite(motion).all():
            raise ValueError("Motion data contains invalid values")

    except (ValueError, OSError) as error:
        raise HTTPException(
            status_code=500,
            detail=f"Unable to load motion data: {error}",
        )

    return {
        "motion_name": safe_name,
        "fps": 30,
        "frames": int(motion.shape[0]),
        "parameters": MOTION_PARAMETERS,
        "data": motion.tolist(),
    }


@app.get("/test-audio/{segment_name}")
def get_test_audio(segment_name: str):
    safe_name = validate_filename(segment_name, ".wav")
    segment_path = TEST_SEGMENT_DIR / safe_name

    if not segment_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Test audio not found",
        )

    return FileResponse(
        segment_path,
        media_type="audio/wav",
        filename=safe_name,
        content_disposition_type="inline",
    )


@app.post("/export-video")
async def create_video(
    recording: UploadFile = File(...),
    audio_name: str = Form(...),
):
    safe_audio_name = validate_filename(
        audio_name,
        ".wav",
    )

    audio_path = TEST_SEGMENT_DIR / safe_audio_name

    if not audio_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Source audio file not found",
        )

    unique_id = uuid4().hex

    recording_path = TEMP_VIDEO_DIR / f"{unique_id}.webm"
    video_name = f"{unique_id}.mp4"
    video_path = OUTPUT_DIR / video_name

    try:
        if recording.content_type not in {
            "video/webm",
            "application/octet-stream",
        }:
            raise HTTPException(
                status_code=400,
                detail="Recording must be a WebM video",
            )

        total_size = 0

        with recording_path.open("wb") as output:
            while chunk := await recording.read(1024 * 1024):
                total_size += len(chunk)

                if total_size > MAX_RECORDING_SIZE:
                    raise HTTPException(
                        status_code=413,
                        detail="Video recording exceeds 150 MB",
                    )

                output.write(chunk)

        if total_size == 0:
            raise HTTPException(
                status_code=400,
                detail="Video recording is empty",
            )

        result = export_mp4(
            recording_path,
            audio_path,
            video_path,
        )

    except HTTPException:
        raise

    except (
        FileNotFoundError,
        RuntimeError,
        OSError,
    ) as error:
        video_path.unlink(missing_ok=True)

        raise HTTPException(
            status_code=500,
            detail=f"Video export failed: {error}",
        )

    finally:
        recording_path.unlink(missing_ok=True)
        await recording.close()

    return {
        "message": "Video exported successfully",
        **result,
        "download_url": f"/download-video/{video_name}",
    }


@app.get("/download-video/{video_name}")
def download_video(video_name: str):
    safe_name = validate_filename(video_name, ".mp4")
    video_path = OUTPUT_DIR / safe_name

    if not video_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Video file not found",
        )

    return FileResponse(
        video_path,
        media_type="video/mp4",
        filename=safe_name,
    )
