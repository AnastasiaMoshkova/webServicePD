"""
Voice router — web endpoints for voice modality.

Follows the same pattern as routers/mimic.py:
    - session isolation via core.session
    - per-session subdirectories for recordings/results
    - POST endpoints return JSON with cookies refreshed
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from core.session import read_or_new_session_id, set_session_cookie
from core.templates import templates

logger = logging.getLogger(__name__)

router = APIRouter()

# =============================================================================
# Пути
# =============================================================================
CONFIG_PATH = Path(__file__).parent.parent / "modalities" / "voice" / "configs" / "voice.yaml"

# Базовые директории (подпапки по session_id создаются на каждый запрос)
RECORDINGS_BASE = "recordings/voice"
RESULTS_BASE = "results/voice"
SPEC_WEB_BASE = "static/results/voice"

os.makedirs(RECORDINGS_BASE, exist_ok=True)
os.makedirs(RESULTS_BASE, exist_ok=True)
os.makedirs(SPEC_WEB_BASE, exist_ok=True)


class ProcessRequest(BaseModel):
    patient_id: str
    exercise: str
    duration: int | None = None


def _session_dirs(session_id: str) -> dict[str, str]:
    """Создаёт и возвращает изолированные директории для сессии."""
    dirs = {
        "recordings": os.path.join(RECORDINGS_BASE, session_id),
        "results": os.path.join(RESULTS_BASE, session_id),
        "spec_web": os.path.join(SPEC_WEB_BASE, session_id),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return dirs


def _reset_session_recordings(session_id: str) -> None:
    """Очистить только записи текущей сессии."""
    d = os.path.join(RECORDINGS_BASE, session_id)
    if os.path.isdir(d):
        shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d, exist_ok=True)


# =============================================================================
# Ленивая загрузка сервиса
# =============================================================================
_voice_service = None


def get_voice_service():
    global _voice_service
    if _voice_service is None:
        from app.modalities.voice.service import VoiceService
        _voice_service = VoiceService(config_path=str(CONFIG_PATH))
    return _voice_service


# =============================================================================
# Эндпоинты
# =============================================================================

@router.get("/voice")
def voice_home(request: Request):
    """Voice analysis page."""
    try:
        service = get_voice_service()
        exercises = service.config.get("exercises", [])
        logger.info(f"Loaded {len(exercises)} exercises from config")
    except Exception as e:
        logger.error(f"Failed to load voice config: {e}")
        exercises = [
            {"value": "a", "label": "Протяжная гласная /а/", "trained": True, "recommended_duration_sec": 5},
            {"value": "phrase", "label": "Чтение фразы (beta)", "trained": False, "recommended_duration_sec": 10},
            {"value": "patak", "label": "Повторение слогов па-та-ка (beta)", "trained": False, "recommended_duration_sec": 10},
        ]

    response = templates.TemplateResponse("voice.html", {
        "request": request,
        "exercises": exercises,
    })
    set_session_cookie(response, read_or_new_session_id(request))
    return response


@router.get("/voice/test")
async def voice_test():
    """Тестовый эндпоинт."""
    return {"status": "voice router works!", "config_path": str(CONFIG_PATH)}


@router.post("/voice/upload")
async def voice_upload(request: Request, file: UploadFile = File(...)):
    """Upload audio file for processing. Сохраняет в подпапку сессии."""
    session_id = read_or_new_session_id(request)
    logger.info(f"[{session_id}] Upload: {file.filename}")

    if file.filename is None:
        raise HTTPException(status_code=400, detail="No filename")

    ext = os.path.splitext(file.filename)[1].lower() or ".wav"
    allowed = {".wav", ".mp3", ".m4a", ".ogg", ".webm", ".flac"}
    if ext not in allowed:
        raise HTTPException(status_code=400, detail=f"Unsupported extension: {ext}")

    _reset_session_recordings(session_id)
    dest = os.path.join(RECORDINGS_BASE, session_id, f"raw_audio{ext}")

    try:
        content = await file.read()
        with open(dest, "wb") as f:
            f.write(content)
        logger.info(f"[{session_id}] Saved {dest} ({len(content)} bytes)")

        response = JSONResponse({
            "status": "success",
            "saved_as": dest,
            "size": len(content),
        })
        set_session_cookie(response, session_id)
        return response
    except Exception as e:
        logger.exception(f"[{session_id}] Upload failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/voice/process")
async def voice_process(request: Request, req: ProcessRequest):
    """Process uploaded audio for this session."""
    session_id = read_or_new_session_id(request)
    logger.info(f"[{session_id}] Process: patient={req.patient_id}, exercise={req.exercise}")

    session_rec = os.path.join(RECORDINGS_BASE, session_id)
    if not os.path.isdir(session_rec):
        raise HTTPException(status_code=400, detail="No audio uploaded for this session.")

    raw_files = [f for f in os.listdir(session_rec) if f.startswith("raw_audio")]
    if not raw_files:
        raise HTTPException(status_code=400, detail="No audio uploaded. Call /voice/upload first.")
    raw_path = os.path.join(session_rec, raw_files[0])

    dirs = _session_dirs(session_id)

    try:
        service = get_voice_service()
        result = service.process(
            audio_path=raw_path,
            patient_id=req.patient_id,
            exercise=req.exercise,
            results_dir=dirs["results"],
            spec_web_dir=dirs["spec_web"],
            spec_url_prefix=f"/static/results/voice/{session_id}",
        )
        logger.info(f"[{session_id}] Processing complete for patient {req.patient_id}")

        response = JSONResponse(content=result)
        set_session_cookie(response, session_id)
        return response

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print("\n" + "=" * 70)
        print("VOICE PROCESS ERROR")
        print("=" * 70)
        print(tb)
        print("=" * 70 + "\n")
        raise HTTPException(status_code=500, detail=f"Processing failed: {e}")