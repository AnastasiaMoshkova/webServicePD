import base64
import json
import os

import aiofiles
import cv2
import numpy as np
from fastapi import APIRouter, File, Form, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from app.modalities.mimic.service import MimicService
from app.routers.utils.FaceMeshModule import FaceMeshDetector
from core.session import get_or_create_session_id, get_session_id_from_websocket, read_or_new_session_id, set_session_cookie
from core.templates import templates

router = APIRouter()
service = MimicService()

# Нейтраль первая
EXERCISES = {
    "neutral": "Нейтральное выражение (10 сек)",
    "smile": "Улыбка с усилием (10 раз)",
    "blink": "Зажмуривание глаз (10 раз)",
    "brows_up": "Поднятие бровей (10 раз)",
    "frown": "Нахмуривание бровей (10 раз)",
    "anger": "Злость",
    "disgust": "Отвращение",
    "fear": "Страх",
    "joy": "Радость",
    "sadness": "Печаль",
    "surprise": "Удивление",
}

local_dir  = "static/recordings_face/"
upload_dir = "static/uploads_face/"


def ensure_dirs(session_id: str):
    """Записи каждой сессии — в своей подпапке, чтобы не пересекаться между пользователями."""
    for base in (local_dir, upload_dir):
        d = os.path.join(base, session_id)
        if not os.path.exists(d):
            os.makedirs(d)


def video_path(session_id: str, mode: str, exercise: str) -> str:
    base = upload_dir if mode == "upload" else local_dir
    return os.path.join(base, session_id, f"{exercise}.avi")


def readb64(uri):
    encoded_data = uri.split(",")[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)


def im_2_b64(image):
    buff = cv2.imencode(".jpg", image)[1]
    return "data:image/jpeg;base64," + base64.b64encode(buff).decode("utf-8")


@router.get("/mimic")
def home(request: Request):
    response = templates.TemplateResponse("mimic.html", {"request": request, "exercises": EXERCISES})
    get_or_create_session_id(request, response)
    return response


@router.post("/mimic/process")
async def process_data(request: Request):
    session_id = read_or_new_session_id(request)
    try:
        data = await request.json()
        exercise = data.get("exercise", "neutral")
        mode = data.get("mode", "camera")

        file_path = video_path(session_id, mode, exercise)
        if not os.path.exists(file_path):
            resp = JSONResponse(
                {"status": "error", "message": "Файл видео не найден. Сначала загрузите или запишите видео."}
            )
            set_session_cookie(resp, session_id)
            return resp

        result = service.process_video(session_id, file_path, exercise_type=exercise)
        resp = JSONResponse({"status": result.get("status", "success"), "data": result})
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        resp = JSONResponse(status_code=500, content={"status": "error", "message": str(e)})
    set_session_cookie(resp, session_id)
    return resp


@router.post("/mimic/upload_video")
async def upload_video(request: Request, file: UploadFile = File(...), exercise: str = Form("smile")):
    session_id = read_or_new_session_id(request)
    ensure_dirs(session_id)
    file_location = video_path(session_id, "upload", exercise)
    try:
        async with aiofiles.open(file_location, "wb") as out_file:
            content = await file.read()
            await out_file.write(content)
        resp = JSONResponse({"status": "success", "message": f"Файл {file.filename} загружен как {exercise}."})
    except Exception as e:
        resp = JSONResponse(status_code=500, content={"status": "error", "message": str(e)})
    set_session_cookie(resp, session_id)
    return resp


@router.websocket("/mimic/ws")
async def websocket_endpoint(websocket: WebSocket):
    session_id = get_session_id_from_websocket(websocket)
    await websocket.accept()
    ensure_dirs(session_id)

    detector = FaceMeshDetector(max_num_faces=1)
    out = None
    current_exercise = None
    frame_count = 0
    DISPLAY_SKIP_FRAMES = 4

    try:
        while True:
            data_text = await websocket.receive_text()
            data_json = json.loads(data_text)

            if "image" not in data_json:
                continue

            img = readb64(data_json["image"])
            frame_count += 1

            should_draw_mesh = data_json.get("draw_mesh", True)
            is_recording = data_json.get("recording", False)
            exercise = data_json.get("exercise", "neutral")

            # detector.findFaceMesh мутирует переданный массив (рисует маску in-place).
            # Нужна отдельная чистая копия для записи в файл.
            img_clean = img.copy()
            img_processed, faces = detector.findFaceMesh(img, draw=should_draw_mesh)

            if is_recording:
                if out is None or current_exercise != exercise:
                    if out is not None:
                        out.release()
                    current_exercise = exercise
                    height, width, _ = img_clean.shape
                    fourcc = cv2.VideoWriter_fourcc(*"XVID")
                    out_path = video_path(session_id, "camera", exercise)
                    out = cv2.VideoWriter(out_path, fourcc, 20.0, (width, height))
                out.write(img_clean)
            else:
                if out is not None:
                    out.release()
                    out = None
                    current_exercise = None

            if frame_count % DISPLAY_SKIP_FRAMES == 0:
                processed_b64 = im_2_b64(img_processed)
                landmarks_data = faces[0] if faces else []
                await websocket.send_text(json.dumps({
                    "image": processed_b64,
                    "landmarks": landmarks_data,
                }))

    except WebSocketDisconnect:
        if out is not None:
            out.release()
        print("Клиент отключился")


@router.post("/mimic/fusion")
async def fusion_endpoint(request: Request):
    session_id = read_or_new_session_id(request)
    try:
        result = service.process_all_fusion(session_id)
        resp = JSONResponse({"status": result.get("status", "success"), "data": result})
    except Exception as e:
        print(f"FUSION ERROR: {e}")
        import traceback
        traceback.print_exc()
        resp = JSONResponse(status_code=500, content={"status": "error", "message": str(e)})
    set_session_cookie(resp, session_id)
    return resp
