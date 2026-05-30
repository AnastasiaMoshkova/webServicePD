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
    return templates.TemplateResponse("hand_tracking.html", {"request": request})


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    service.reset_local_dir()
    await websocket.accept()
    logger.info("WebSocket подключен")
    detector = htm.handDetector(detectionCon=0.7)

    try:
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                break

            data = message.get("bytes")
            if not data:
                continue

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
            if service.recording:
                # инициализация записи
                if service.out is None:
                    service.init_writer(frame_orig.shape)

                # вычисляем, сколько осталось секунд
                if service.start_time and service.countdown_time:
                    elapsed = (datetime.now() - service.start_time).total_seconds()
                    remaining = int(service.countdown_time - elapsed)

                    # если время вышло — автоостановка
                    if remaining <= 0:
                        service.stop_record(clear_output=False)
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
                if service.out is not None:
                    try:
                        service.out.write(frame_orig)
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
        service.stop_record(clear_output=False)


@router.post("/start_record")
async def start_record(duration: int = 20):  # по умолчанию 20 секунд
    """Запуск записи с обратным отсчётом"""
    if service.recording:
        return JSONResponse({"status": "already recording"})
    service.start_record(duration)

    return JSONResponse({"status": "started", "duration": duration})


@router.post("/stop_record")
async def stop_record():
    """Остановка записи"""
    saved_file = service.stop_record(clear_output=True)
    return JSONResponse({"status": "stopped", "saved_file": saved_file})


@router.post("/predict_binary")
async def predict_binary(payload: Dict[str, Any] = Body(...)):
    try:
        result = service.predict_binary(payload)
        return JSONResponse(content=result)
    except NotImplementedError as e:
        return JSONResponse(status_code=501, content={"status": "error", "message": str(e)})


@router.post("/upload")
async def upload(
    file: UploadFile = File(...),
):
    try:
        content = await file.read()
        result = service.upload(file.filename, content)
        return JSONResponse(content=result)
    except ValueError as e:
        return JSONResponse(status_code=400, content={"status": "error", "message": str(e)})
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": f"Failed to upload file: {str(e)}"},
        )


@router.post("/raw_data_processing")
async def raw_data_processing(experiment_info: RawDataRequest):
    result = service.raw_data_processing(experiment_info.model_dump())
    return JSONResponse(content=result)


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
