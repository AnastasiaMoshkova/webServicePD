from fastapi import APIRouter, File, Request, UploadFile
from fastapi.responses import JSONResponse

from app.modalities.gait.service import GaitService
from core.session import get_or_create_session_id, read_or_new_session_id, set_session_cookie
from core.templates import templates

router = APIRouter()
service = GaitService()


@router.get("/gait")
def home(request: Request):
    response = templates.TemplateResponse("gait.html", {"request": request})
    get_or_create_session_id(request, response)
    return response


@router.get("/api/models-status")
def models_status():
    return service.models_status()


@router.post("/api/process-signal")
async def process_signal(
    request: Request,
    acc_file: UploadFile = File(...),
    gyro_file: UploadFile = File(...),
):
    session_id = read_or_new_session_id(request)
    try:
        acc_content = await acc_file.read()
        gyro_content = await gyro_file.read()
        result = service.process_files(session_id, acc_content, gyro_content)
        resp = JSONResponse(content=result)
    except ValueError as exc:
        resp = JSONResponse(status_code=400, content={"status": "error", "message": str(exc)})
    except Exception as exc:
        resp = JSONResponse(status_code=500, content={"status": "error", "message": str(exc)})
    set_session_cookie(resp, session_id)
    return resp


@router.post("/api/recalculate")
async def recalculate(request: Request):
    session_id = read_or_new_session_id(request)
    try:
        payload = await request.json()
        points = payload.get("points", {})
        result = service.recalculate(session_id, points)
        resp = JSONResponse(content=result)
    except ValueError as exc:
        resp = JSONResponse(status_code=400, content={"status": "error", "message": str(exc)})
    except Exception as exc:
        resp = JSONResponse(status_code=500, content={"status": "error", "message": str(exc)})
    set_session_cookie(resp, session_id)
    return resp
