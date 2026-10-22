
from pathlib import Path
from uuid import uuid4

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from backend.audio_utils import (
    convert_to_standard_wav,
    create_test_segment,
    get_audio_metadata,
)
from backend.facial_motion import generate_facial_motion
from backend.motion_generator import generate_motion
from backend.motion_smoothing import generate_smoothed_motion
from backend.speech_features import extract_speech_features
from backend.speech_visemes import extract_visemes
from backend.video_export import export_mp4
from backend.viseme_motion import generate_viseme_motion
from backend.voice_refinement import refine_voice


app = FastAPI(
    title="Digital Human Animation API",
    version="1.3.0",
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
VISEME_DIR = BASE_DIR / "data" / "visemes"
OUTPUT_DIR = BASE_DIR / "outputs"
TEMP_VIDEO_DIR = OUTPUT_DIR / "temporary"

for directory in [
    UPLOAD_DIR,
    PROCESSED_DIR,
    FEATURE_DIR,
    TEST_SEGMENT_DIR,
    MOTION_DIR,
    VISEME_DIR,
    OUTPUT_DIR,
    TEMP_VIDEO_DIR,
]:
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

ALLOWED_AUDIO_TYPES = {
    "audio/mpeg",
    "audio/mp3",
    "audio/wav",
    "audio/x-wav",
    "audio/wave",
    "audio/mp4",
    "audio/x-m4a",
    "audio/aac",
    "application/octet-stream",
}

ALLOWED_EXTENSIONS = {
    ".mp3",
    ".wav",
    ".m4a",
    ".aac",
}

MAX_FILE_SIZE = 25 * 1024 * 1024
MAX_RECORDING_SIZE = 150 * 1024 * 1024
PROCESSING_DURATION = 15
FPS = 30

MOTION_PARAMETERS = [
    "mouth_open",
    "mouth_width",
    "brow_raise",
    "head_yaw",
    "head_pitch",
    "head_roll",
    "gesture_intensity",
]

FACIAL_PARAMETERS = [
    *MOTION_PARAMETERS,
    "lip_closure",
    "lip_rounding",
    "lip_stretch",
    "lower_lip_contact",
]


def validate_filename(
    filename: str,
    suffix: str,
) -> str:
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


def motion_response(path: Path) -> dict:
    motion = np.load(
        path,
        allow_pickle=False,
    )

    if (
        motion.ndim != 2
        or motion.shape[0] == 0
        or motion.shape[1] not in {7, 11}
    ):
        raise ValueError(
            "Motion must have 7 or 11 channels"
        )

    if not np.isfinite(motion).all():
        raise ValueError(
            "Motion contains invalid values"
        )

    parameters = (
        FACIAL_PARAMETERS
        if motion.shape[1] == 11
        else MOTION_PARAMETERS
    )

    return {
        "motion_name": path.name,
        "fps": FPS,
        "frames": int(motion.shape[0]),
        "parameters": parameters,
        "data": motion.tolist(),
    }


@app.get("/")
def root():
    return {
        "message": "Digital Human Animation API is running"
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/upload-audio")
async def upload_audio(
    file: UploadFile = File(...),
):
    try:
        original_name = Path(
            file.filename or ""
        ).name

        extension = Path(
            original_name
        ).suffix.lower()

        if extension not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail="Unsupported file extension",
            )

        content_type = (
            (file.content_type or "")
            .split(";")[0]
            .strip()
            .lower()
        )

        if content_type not in ALLOWED_AUDIO_TYPES:
            raise HTTPException(
                status_code=400,
                detail="Unsupported audio format",
            )

        content = await file.read(
            MAX_FILE_SIZE + 1
        )

        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail="Audio file must not exceed 25 MB",
            )

        if not content:
            raise HTTPException(
                status_code=400,
                detail="Audio file is empty",
            )

        unique_id = uuid4().hex

        stored_name = (
            f"{unique_id}{extension}"
        )

        processed_name = (
            f"{unique_id}.wav"
        )

        uploaded_path = (
            UPLOAD_DIR / stored_name
        )

        processed_path = (
            PROCESSED_DIR / processed_name
        )

        uploaded_path.write_bytes(content)

        try:
            convert_to_standard_wav(
                uploaded_path,
                processed_path,
            )

            metadata = get_audio_metadata(
                processed_path
            )

        except Exception as error:
            uploaded_path.unlink(
                missing_ok=True
            )

            processed_path.unlink(
                missing_ok=True
            )

            raise HTTPException(
                status_code=422,
                detail=(
                    f"Audio preprocessing failed: {error}"
                ),
            ) from error

        return {
            "message": "Audio uploaded and processed successfully",
            "original_name": original_name,
            "stored_name": stored_name,
            "processed_name": processed_name,
            "size": len(content),
            "content_type": content_type,
            "metadata": metadata,
        }

    finally:
        await file.close()


@app.post("/create-test-segment/{processed_name}")
def create_short_test_segment(
    processed_name: str,
):
    safe_name = validate_filename(
        processed_name,
        ".wav",
    )

    processed_path = (
        PROCESSED_DIR / safe_name
    )

    if not processed_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Processed audio file not found",
        )

    segment_name = (
        f"{processed_path.stem}_{PROCESSING_DURATION}s.wav"
    )

    segment_path = (
        TEST_SEGMENT_DIR / segment_name
    )

    try:
        create_test_segment(
            processed_path,
            segment_path,
            duration_seconds=PROCESSING_DURATION,
        )

        metadata = get_audio_metadata(
            segment_path
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Test segment creation failed: {error}",
        ) from error

    return {
        "message": "Test segment created successfully",
        "segment_name": segment_name,
        "metadata": metadata,
    }


@app.post("/extract-features/{processed_name}")
def extract_features(
    processed_name: str,
):
    safe_name = validate_filename(
        processed_name,
        ".wav",
    )

    processed_path = (
        PROCESSED_DIR / safe_name
    )

    if not processed_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Processed audio file not found",
        )

    feature_path = (
        FEATURE_DIR
        / f"{processed_path.stem}.npy"
    )

    try:
        result = extract_speech_features(
            processed_path,
            feature_path,
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Speech feature extraction failed: {error}",
        ) from error

    return {
        "message": "Speech features extracted successfully",
        "processed_name": safe_name,
        **result,
    }


@app.post("/generate-motion/{segment_name}")
def generate_motion_data(
    segment_name: str,
):
    safe_name = validate_filename(
        segment_name,
        ".wav",
    )

    segment_path = (
        TEST_SEGMENT_DIR / safe_name
    )

    if not segment_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Test segment not found",
        )

    motion_path = (
        MOTION_DIR
        / f"{segment_path.stem}_motion.npy"
    )

    try:
        result = generate_motion(
            segment_path,
            motion_path,
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Motion generation failed: {error}",
        ) from error

    return {
        "message": "Motion generated successfully",
        "segment_name": safe_name,
        **result,
    }


@app.post("/process-audio/{processed_name}")
def process_audio(
    processed_name: str,
    speed: float = Query(
        default=0.9,
        ge=0.75,
        le=1.0,
    ),
):
    safe_name = validate_filename(
        processed_name,
        ".wav",
    )

    processed_path = (
        PROCESSED_DIR / safe_name
    )

    if not processed_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Processed audio file not found",
        )

    speed_tag = (
        f"speed{round(speed * 100):03d}"
    )

    stem = (
        f"{processed_path.stem}_"
        f"{PROCESSING_DURATION}s_{speed_tag}"
    )

    source_segment_path = (
        TEST_SEGMENT_DIR
        / f"{processed_path.stem}_{PROCESSING_DURATION}s_source.wav"
    )

    segment_path = (
        TEST_SEGMENT_DIR / f"{stem}.wav"
    )

    feature_path = (
        FEATURE_DIR / f"{stem}_features.npy"
    )

    viseme_path = (
        VISEME_DIR / f"{stem}_visemes.json"
    )

    baseline_path = (
        MOTION_DIR / f"{stem}_motion.npy"
    )

    enhanced_path = (
        MOTION_DIR / f"{stem}_viseme_motion.npy"
    )

    smoothed_path = (
        MOTION_DIR / f"{stem}_smoothed_motion.npy"
    )

    facial_enhanced_path = (
        MOTION_DIR
        / f"{stem}_facial_viseme_motion.npy"
    )

    facial_smoothed_path = (
        MOTION_DIR
        / f"{stem}_facial_smoothed_motion.npy"
    )

    generated_paths = [
        segment_path,
        feature_path,
        viseme_path,
        baseline_path,
        enhanced_path,
        smoothed_path,
        facial_enhanced_path,
        facial_smoothed_path,
    ]

    try:
        create_test_segment(
            processed_path,
            source_segment_path,
            duration_seconds=PROCESSING_DURATION,
        )

        if speed == 1.0:
            segment_path.write_bytes(
                source_segment_path.read_bytes()
            )

            voice_details = {
                "speed": 1.0,
                "pitch_preserving": True,
            }

        else:
            voice_details = refine_voice(
                source_segment_path,
                segment_path,
                speed,
            )

        audio_metadata = get_audio_metadata(
            segment_path
        )

        baseline_result = generate_motion(
            segment_path,
            baseline_path,
        )

        feature_result = extract_speech_features(
            segment_path,
            feature_path,
        )

        viseme_result = extract_visemes(
            segment_path,
            viseme_path,
        )

        enhanced_result = generate_viseme_motion(
            baseline_path,
            viseme_path,
            enhanced_path,
        )

        smoothing_result = generate_smoothed_motion(
            enhanced_path,
            smoothed_path,
        )

        facial_enhanced_result = generate_facial_motion(
            enhanced_path,
            viseme_path,
            facial_enhanced_path,
        )

        facial_smoothed_result = generate_facial_motion(
            smoothed_path,
            viseme_path,
            facial_smoothed_path,
        )

        baseline = np.load(
            baseline_path,
            allow_pickle=False,
        )

        enhanced = np.load(
            facial_enhanced_path,
            allow_pickle=False,
        )

        smoothed = np.load(
            facial_smoothed_path,
            allow_pickle=False,
        )

        if (
            baseline.ndim != 2
            or baseline.shape[1] != 7
            or enhanced.ndim != 2
            or enhanced.shape[1] != 11
            or smoothed.ndim != 2
            or smoothed.shape[1] != 11
            or baseline.shape[0] != enhanced.shape[0]
            or baseline.shape[0] != smoothed.shape[0]
            or not np.isfinite(baseline).all()
            or not np.isfinite(enhanced).all()
            or not np.isfinite(smoothed).all()
        ):
            raise ValueError(
                "Generated facial motion sequences are inconsistent"
            )

    except Exception as error:
        for path in generated_paths:
            path.unlink(
                missing_ok=True
            )

        raise HTTPException(
            status_code=500,
            detail=f"Audio processing failed: {error}",
        ) from error

    finally:
        source_segment_path.unlink(
            missing_ok=True
        )

    return {
        "message": "Audio animation processing completed",
        "processed_name": safe_name,
        "audio_name": segment_path.name,
        "audio_url": f"/test-audio/{segment_path.name}",
        "audio_metadata": audio_metadata,
        "voice_speed": speed,
        "voice_details": voice_details,
        "fps": FPS,
        "frames": int(baseline.shape[0]),
        "duration": round(
            baseline.shape[0] / FPS,
            3,
        ),
        "motion_files": {
            "baseline": baseline_path.name,
            "viseme": facial_enhanced_path.name,
            "smoothed": facial_smoothed_path.name,
        },
        "motion_urls": {
            "baseline": (
                f"/motion-data/{baseline_path.name}"
            ),
            "viseme": (
                f"/motion-data/{facial_enhanced_path.name}"
            ),
            "smoothed": (
                f"/motion-data/{facial_smoothed_path.name}"
            ),
        },
        "feature_file": feature_path.name,
        "viseme_file": viseme_path.name,
        "transcription": viseme_result["transcription"],
        "viseme_segments": len(
            viseme_result["segments"]
        ),
        "processing_details": {
            "baseline": baseline_result,
            "features": feature_result,
            "enhanced": enhanced_result,
            "smoothing": smoothing_result,
            "facial_enhanced": facial_enhanced_result,
            "facial_smoothed": facial_smoothed_result,
        },
        "note": (
            "Facial controls use approximate phoneme timing. "
            "Lip-sync accuracy has not been validated."
        ),
    }


@app.get("/motion-data/{motion_name}")
def get_motion_data(
    motion_name: str,
):
    safe_name = validate_filename(
        motion_name,
        ".npy",
    )

    motion_path = (
        MOTION_DIR / safe_name
    )

    if not motion_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Motion file not found",
        )

    try:
        return motion_response(
            motion_path
        )

    except (ValueError, OSError) as error:
        raise HTTPException(
            status_code=500,
            detail=f"Unable to load motion data: {error}",
        ) from error


@app.get("/test-audio/{segment_name}")
def get_test_audio(
    segment_name: str,
):
    safe_name = validate_filename(
        segment_name,
        ".wav",
    )

    segment_path = (
        TEST_SEGMENT_DIR / safe_name
    )

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

    audio_path = (
        TEST_SEGMENT_DIR / safe_audio_name
    )

    if not audio_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Source audio file not found",
        )

    unique_id = uuid4().hex

    recording_path = (
        TEMP_VIDEO_DIR / f"{unique_id}.webm"
    )

    video_name = (
        f"{unique_id}.mp4"
    )

    video_path = (
        OUTPUT_DIR / video_name
    )

    try:
        recording_type = (
            (recording.content_type or "")
            .split(";")[0]
            .strip()
            .lower()
        )

        if recording_type not in {
            "video/webm",
            "application/octet-stream",
        }:
            raise HTTPException(
                status_code=400,
                detail="Recording must be a WebM video",
            )

        total_size = 0

        with recording_path.open("wb") as output:
            while chunk := await recording.read(
                1024 * 1024
            ):
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
        video_path.unlink(
            missing_ok=True
        )
        raise

    except (
        FileNotFoundError,
        RuntimeError,
        OSError,
    ) as error:
        video_path.unlink(
            missing_ok=True
        )

        raise HTTPException(
            status_code=500,
            detail=f"Video export failed: {error}",
        ) from error

    finally:
        recording_path.unlink(
            missing_ok=True
        )

        await recording.close()

    return {
        "message": "Video exported successfully",
        **result,
        "download_url": (
            f"/download-video/{video_name}"
        ),
    }


@app.get("/download-video/{video_name}")
def download_video(
    video_name: str,
):
    safe_name = validate_filename(
        video_name,
        ".mp4",
    )

    video_path = (
        OUTPUT_DIR / safe_name
    )

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
