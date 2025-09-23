import base64
import io
import logging
import os
import shutil
import cv2
import numpy as np
from pydantic import BaseModel
from envyaml import EnvYAML
from typing import Dict
from fastapi import APIRouter, File, Form, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, RedirectResponse

import HandTrackingModule as htm
from core.minio_client import BUCKET_NAME, minio_client
from core.templates import templates
from hand_processing.automarking import AutoMarking
from hand_processing.raw_data_processing import PreProcessing

from hand_processing.feature_extraction import FE

# from hand_processing.statistic import Statistic

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
router = APIRouter()
_fe_config = None
recording = False
out = None
output_file = None
wCam, hCam = 640, 480
local_dir = "/app/static/recordings/"
shutil.rmtree(local_dir, ignore_errors=True)
if not os.path.isdir(local_dir):
    os.mkdir(local_dir)


class RawDataRequest(BaseModel):
    patientId: str
    exercise: str
    confidence: float


def get_fe_config() -> Dict:
    """Полный конфиг"""
    global _fe_config
    if _fe_config is None:
        _fe_config = dict(EnvYAML("configs/feature.yaml"))
    return _fe_config


feature_config = get_fe_config()
hand_data_processing = PreProcessing()
automarker = AutoMarking()
feature_extraction = FE(feature_config)
# statistic = Statistic()


@router.get("/")
async def hand_tracking_redirect():
    return RedirectResponse(url="/hand_tracking", status_code=301)


@router.get("/hand_tracking")
def home(request: Request):
    return templates.TemplateResponse("hand_tracking.html", {"request": request})


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global recording, out, output_file
    await websocket.accept()
    print("✅ WebSocket подключен")
    detector = htm.handDetector(detectionCon=0.7)

    try:
        while True:
            data = await websocket.receive_text()
            if not data or len(data) == 0:
                print("Получена пустая строка")
                continue
            try:
                img_bytes = base64.b64decode(data, validate=True)
            except Exception as e:
                print(f"Ошибка декодирования base64: {e}")
                continue
            np_arr = np.frombuffer(img_bytes, np.uint8)
            if np_arr.size == 0:
                print("Пустой numpy массив")
                continue
            frame_orig = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame_orig is None:
                print("Не удалось декодировать изображение (битые данные)")
                continue

            # Обработка через MediaPipe
            frame = detector.findHands(frame_orig)
            lmList = detector.findPosition(frame_orig, draw=True)

            # Пример: расстояние между указательным и большим пальцами
            if len(lmList) >= 9:
                x1, y1 = lmList[4][1], lmList[4][2]
                x2, y2 = lmList[8][1], lmList[8][2]
                length = int(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5)
                cv2.putText(
                    frame,
                    f"Dist: {length}px",
                    (30, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )

            if recording and out is not None:
                out.write(frame_orig)

            success, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if not success:
                continue

            jpg_as_text = "data:image/jpeg;base64," + base64.b64encode(buffer).decode("utf-8")
            await websocket.send_text(jpg_as_text)

    except WebSocketDisconnect:
        print("Клиент отключился")
    except Exception as e:
        print(f"Ошибка в WebSocket: {e}")
    finally:
        if out is not None:
            out.release()
            out = None
        recording = False


@router.post("/start_record")
async def start_record():
    """Запуск записи"""
    global recording, out, output_file
    if recording:
        return JSONResponse({"status": "already recording"})

    output_file = os.path.join(local_dir, "test_video.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(output_file, fourcc, 20.0, (wCam, hCam))
    recording = True
    return JSONResponse({"status": "started"})


@router.post("/stop_record")
async def stop_record():
    """Остановка записи"""
    global recording, out, output_file
    recording = False
    if out is not None:
        out.release()
        out = None
    saved_file = output_file
    output_file = None
    return JSONResponse({"status": "stopped", "saved_file": saved_file})


@router.post("/upload")
async def upload(
    file: UploadFile = File(...),
):
    rel_path = file.filename
    logger.info(f"Received file: {rel_path}")
    if not minio_client:
        return JSONResponse(
            status_code=500, content={"status": "error", "message": "MinIO client not initialized"}
        )

    # Очищаем и пересоздаём локальную директорию
    shutil.rmtree(local_dir, ignore_errors=True)
    if not os.path.isdir(local_dir):
        os.mkdir(local_dir)

    try:
        rel_path = rel_path.strip().lstrip("/")
        print(f"Parsed path: {rel_path}")
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": f"Invalid path: {str(e)}"},
        )

    try:
        if not minio_client.bucket_exists(BUCKET_NAME):
            minio_client.make_bucket(BUCKET_NAME)
            print(f"Bucket {BUCKET_NAME} created")
    except Exception as e:
        print(f"MinIO connection error: {e}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": f"MinIO connection failed: {str(e)}"},
        )

    try:
        content = await file.read()
        local_path = os.path.join(local_dir, "test_video.mp4")
        with open(local_path, "wb") as f:
            f.write(content)
        print(f"Saved locally: {local_path}")

        minio_client.put_object(
            BUCKET_NAME,
            rel_path,
            io.BytesIO(content),
            length=len(content),
            content_type=file.content_type or "application/octet-stream",
        )
        print(f"Successfully uploaded: {rel_path}")

        return {
            "status": "success",
            "uploaded": rel_path,
        }

    except Exception as e:
        print(f"Failed to upload {file.filename}: {e}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": f"Failed to upload file: {str(e)}"},
        )


@router.post("/raw_data_processing")
async def raw_data_processing(experiment_info: RawDataRequest):

    patient_id = experiment_info.patientId
    exercise = experiment_info.exercise
    confidence = experiment_info.confidence
    logger.info(
        f"Получены данные: patientId={patient_id}, exercise={exercise}, confidence={confidence}"
    )
    # download_folder(BUCKET_NAME, patient_folder, local_dir)
    fps = hand_data_processing.processing(local_dir)
    maxP, minP, maxA, minA, values, frames = automarker.processing(local_dir, exercise, fps)
    features, features_norm = feature_extraction.processing(
        os.path.join(local_dir, "auto_algoritm_MP"), exercise
    )
    # stats = statistic.processing()
    logger.info(f"Получены признаки: {features}")
    logger.info(f"Нормы: {features_norm}")
    image_signal_path = "/static/results/images/signal_picture.png"
    # image_stats_path = "/static/results/mp1_L_m1__mp_angle.png"
    timestamps = np.array(frames) / fps
    logger.info(f"timestmps: {timestamps}")
    result = {
        "values": values.tolist() if hasattr(values, "tolist") else values,
        "frames": frames.tolist() if hasattr(frames, "tolist") else frames,
        "times": timestamps.tolist() if hasattr(timestamps, "tolist") else timestamps,
        "max_X": maxP.tolist() if hasattr(maxP, "tolist") else maxP,
        "min_X": minP.tolist() if hasattr(minP, "tolist") else minP,
        "max_Y": maxA.tolist() if hasattr(maxA, "tolist") else maxA,
        "min_Y": minA.tolist() if hasattr(minA, "tolist") else minA,
        "features": features_norm,
        # "signal_img": image_signal_path,
        # "stats_img": image_stats_path,
    }

    return JSONResponse(content=result)


def download_folder(BUCKET_NAME: str, prefix: str, local_dir: str):
    """
    Скачивает папку из MinIO, сохраняя структуру директорий.

    :param BUCKET_NAME: имя бакета
    :param prefix: префикс (папка) в бакете, которую нужно скачать
    :param dest_dir: локальная папка для сохранения
    """
    os.makedirs(local_dir, exist_ok=True)
    objects = minio_client.list_objects(BUCKET_NAME, prefix=prefix, recursive=True)

    for obj in objects:
        logger.info("tute")
        relative_path = obj.object_name[len(prefix) :].lstrip("/")  # убираем префикс и лишний слэш
        local_path = os.path.join(local_dir, "patinet1", relative_path)

        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        # Скачиваем объект
        minio_client.fget_object(BUCKET_NAME, obj.object_name, local_path)
        logger.info(f"✅ Скачан: {obj.object_name} -> {local_path}")
