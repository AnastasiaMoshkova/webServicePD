import logging
from datetime import datetime
from typing import Any, Dict

import cv2
import numpy as np
from fastapi import APIRouter, Body, File, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel

import app.routers.utils.HandTrackingModule as htm
from app.modalities.hand.service import HandTrackingService
from core.session import get_or_create_session_id, get_session_id_from_websocket, read_or_new_session_id, set_session_cookie
from core.templates import templates

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
router = APIRouter()
service = HandTrackingService()


class RawDataRequest(BaseModel):
    patientId: str
    exercise: str
    confidence: float
    hand: str


@router.get("/")
async def hand_tracking_redirect():
    return RedirectResponse(url="/hand_tracking", status_code=301)


@router.get("/hand_tracking")
def home(request: Request):
    response = templates.TemplateResponse("hand_tracking.html", {"request": request})
    get_or_create_session_id(request, response)
    return response


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    session_id = get_session_id_from_websocket(websocket)
    service.reset_local_dir(session_id)
    await websocket.accept()
    logger.info("WebSocket подключен (session=%s)", session_id)
    detector = htm.handDetector(detectionCon=0.7)
    frame_count = 0
    # MediaPipe Hands на кадр ~100+ мс — заметно дольше, чем интервал между
    # кадрами с камеры. Клиент теперь ждёт ответа на каждый отправленный кадр
    # (ack-based throttling, см. static/js/hand_tracking.js), поэтому здесь
    # отвечаем на каждый обработанный кадр — темп естественно ограничивается
    # реальной скоростью обработки, без искусственного пропуска кадров.

    try:
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                break

            data = message.get("bytes")
            if not data:
                continue

            frame_count += 1
            np_arr = np.frombuffer(data, np.uint8)
            if np_arr.size == 0:
                logger.error("Пустой numpy массив")
                continue
            frame_orig = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame_orig is None:
                logger.error("Не удалось декодировать изображение (битые данные)")
                continue

            # Обработка через MediaPipe
            frame = detector.findHands(frame_orig.copy())
            s = service.get_session(session_id)
            if s["recording"]:
                # инициализация записи
                if s["out"] is None:
                    service.init_writer(session_id, frame_orig.shape)

                # вычисляем, сколько осталось секунд
                if s["start_time"] and s["countdown_time"]:
                    elapsed = (datetime.now() - s["start_time"]).total_seconds()
                    remaining = int(s["countdown_time"] - elapsed)

                    # если время вышло — автоостановка
                    if remaining <= 0:
                        service.stop_record(session_id, clear_output=False)
                        remaining = 0
                        try:
                            await websocket.send_json(
                                {"type": "recording_stopped", "reason": "time_up"}
                            )
                            logger.info("JSON отправлен фронту: recording_stopped")
                        except Exception as e:
                            logger.info("Не удалось отправить JSON:", e)

                    # рисуем таймер
                    cv2.putText(
                        frame,
                        f"{remaining} c",
                        (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.5,
                        (0, 0, 255),
                        3,
                    )

                # записываем только если out реально открыт
                if s["out"] is not None:
                    try:
                        s["out"].write(frame_orig)
                    except Exception as e:
                        print("Ошибка записи кадра:", e)

            success, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 45])
            if not success:
                continue
            await websocket.send_bytes(buffer.tobytes())

    except WebSocketDisconnect:
        print("Клиент отключился")
    except Exception as e:
        print(f"Ошибка в WebSocket: {e}")
    finally:
        service.stop_record(session_id, clear_output=False)


@router.post("/start_record")
async def start_record(request: Request, duration: int = 20):  # по умолчанию 20 секунд
    """Запуск записи с обратным отсчётом"""
    session_id = read_or_new_session_id(request)
    if service.get_session(session_id)["recording"]:
        resp = JSONResponse({"status": "already recording"})
    else:
        service.start_record(session_id, duration)
        resp = JSONResponse({"status": "started", "duration": duration})
    set_session_cookie(resp, session_id)
    return resp


@router.post("/stop_record")
async def stop_record(request: Request):
    """Остановка записи"""
    session_id = read_or_new_session_id(request)
    saved_file = service.stop_record(session_id, clear_output=True)
    resp = JSONResponse({"status": "stopped", "saved_file": saved_file})
    set_session_cookie(resp, session_id)
    return resp


@router.post("/predict_binary")
async def predict_binary(payload: Dict[str, Any] = Body(...)):
    try:
        result = service.predict_binary(payload)
        return JSONResponse(content=result)
    except NotImplementedError as e:
        return JSONResponse(status_code=501, content={"status": "error", "message": str(e)})


@router.post("/upload")
async def upload(request: Request, file: UploadFile = File(...)):
    session_id = read_or_new_session_id(request)
    try:
        content = await file.read()
        result = service.upload(session_id, file.filename, content)
        resp = JSONResponse(content=result)
    except ValueError as e:
        resp = JSONResponse(status_code=400, content={"status": "error", "message": str(e)})
    except Exception as e:
        resp = JSONResponse(
            status_code=500,
            content={"status": "error", "message": f"Failed to upload file: {str(e)}"},
        )
    set_session_cookie(resp, session_id)
    return resp


@router.post("/raw_data_processing")
async def raw_data_processing(request: Request, experiment_info: RawDataRequest):
    session_id = read_or_new_session_id(request)
    try:
        result = service.raw_data_processing(session_id, experiment_info.model_dump())
        resp = JSONResponse(content=result)
    except ValueError as e:
        resp = JSONResponse(status_code=400, content={"status": "error", "message": str(e)})
    except Exception as e:
        logger.exception("Ошибка обработки данных руки")
        resp = JSONResponse(
            status_code=500,
            content={"status": "error", "message": f"Ошибка обработки: {e}"},
        )
    set_session_cookie(resp, session_id)
    return resp


@router.post("/insert_in_db")
async def insert_in_db(payload: Dict[str, Any] = Body(...)):
    try:
        result = service.insert_in_db(payload)
        return JSONResponse(content=result)
    except NotImplementedError as e:
        return JSONResponse(status_code=501, content={"status": "error", "message": str(e)})


# def download_folder(BUCKET_NAME: str, prefix: str, local_dir: str):
#     """
#     Скачивает папку из MinIO, сохраняя структуру директорий.

#     :param BUCKET_NAME: имя бакета
#     :param prefix: префикс (папка) в бакете, которую нужно скачать
#     :param dest_dir: локальная папка для сохранения
#     """
#     os.makedirs(local_dir, exist_ok=True)
#     objects = minio_client.list_objects(BUCKET_NAME, prefix=prefix, recursive=True)

#     for obj in objects:
#         logger.info("tute")
#         relative_path = obj.object_name[len(prefix) :].lstrip("/")  # убираем префикс и лишний слэш
#         local_path = os.path.join(local_dir, "patinet1", relative_path)

#         os.makedirs(os.path.dirname(local_path), exist_ok=True)

#         # Скачиваем объект
#         minio_client.fget_object(BUCKET_NAME, obj.object_name, local_path)
#         logger.info(f"✅ Скачан: {obj.object_name} -> {local_path}")
