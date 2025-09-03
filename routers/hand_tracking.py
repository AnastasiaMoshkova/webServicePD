import cv2
import base64
import numpy as np
import time
import os
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import JSONResponse
import HandTrackingModule as htm  # твой модуль
from core.templates import templates
import asyncio

router = APIRouter()

# глобальные переменные
recording = False
out = None
output_file = None
wCam, hCam = 640, 480


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

            # Проверка: не пустая ли строка
            if not data or len(data) == 0:
                print("⚠️ Получена пустая строка")
                continue

            try:
                # Декодируем base64
                img_bytes = base64.b64decode(
                    data, validate=True
                )  # validate=True — проверяет корректность
            except Exception as e:
                print(f"❌ Ошибка декодирования base64: {e}")
                continue

            # Проверка: не пустые ли байты
            if len(img_bytes) == 0:
                print("⚠️ Пустые байты изображения")
                continue

            # Конвертируем в numpy
            np_arr = np.frombuffer(img_bytes, np.uint8)
            # Проверка: не пустой ли массив
            if np_arr.size == 0:
                print("⚠️ Пустой numpy массив")
                continue

            # Декодируем изображение
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is None:
                print("⚠️ Не удалось декодировать изображение (битые данные)")
                continue

            # Обработка через MediaPipe
            frame = detector.findHands(frame)
            lmList = detector.findPosition(frame, draw=True)

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

            # Запись видео
            if recording and out is not None:
                out.write(frame)

            # Кодируем и отправляем обратно
            success, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if not success:
                continue

            jpg_as_text = "data:image/jpeg;base64," + base64.b64encode(buffer).decode("utf-8")
            await websocket.send_text(jpg_as_text)

    except WebSocketDisconnect:
        print("🔌 Клиент отключился")
    except Exception as e:
        print(f"❌ Ошибка в WebSocket: {e}")
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

    os.makedirs("recordings", exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = f"recordings/hand_tracking_{timestamp}.avi"
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
