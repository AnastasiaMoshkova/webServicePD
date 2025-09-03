import numpy as np
import os
from fastapi import APIRouter, Request, UploadFile, Form, File
from fastapi.responses import JSONResponse
import HandTrackingModule as htm
from core.templates import templates
from minio import Minio
import io
import os
import json
import socket
import requests


# Проверка доступности MinIO
def check_minio_connection():
    try:
        # Проверяем, можем ли разрешить имя
        ip = socket.gethostbyname("pd-minio")
        print(f"Resolved pd_minio to IP: {ip}")

        # Проверяем доступность порта
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        result = sock.connect_ex((ip, 9000))
        if result == 0:
            print("MinIO port 9000 is open")
        else:
            print("MinIO port 9000 is closed")
        sock.close()

        return True
    except Exception as e:
        print(f"Connection check failed: {e}")
        return False


router = APIRouter()

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "pd-minio:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "admin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "123admin123")
print(f"Current MINIO_ENDPOINT: {MINIO_ENDPOINT}")


# Инициализация MinIO клиента
try:
    minio_client = Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=False,
    )
    print(f"MinIO client initialized with endpoint: {MINIO_ENDPOINT}")
except Exception as e:
    print(f"Failed to initialize MinIO client: {e}")
    minio_client = None

check_minio_connection()
bucket_name = "patients"


@router.get("/files_processing")
def home(request: Request):
    return templates.TemplateResponse("files_processing.html", {"request": request})


@router.post("/upload")
async def upload(
    files: list[UploadFile] = File(...),
    paths_json: str = Form(...),
):
    print(f"Received {len(files)} files")

    if not minio_client:
        return JSONResponse(
            status_code=500, content={"status": "error", "message": "MinIO client not initialized"}
        )

    try:
        # Парсим JSON строку в список путей
        paths = json.loads(paths_json)
        print(f"Parsed {len(paths)} paths from JSON")
    except json.JSONDecodeError as e:
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": f"Invalid JSON format for paths: {str(e)}"},
        )

    # Проверяем соответствие количества файлов и путей
    if len(files) != len(paths):
        return JSONResponse(
            status_code=400,
            content={
                "status": "error",
                "message": f"Number of files ({len(files)}) and paths ({len(paths)}) must match",
            },
        )

    # Остальной код без изменений
    try:
        if not minio_client.bucket_exists(bucket_name):
            minio_client.make_bucket(bucket_name)
            print(f"Bucket {bucket_name} created")
    except Exception as e:
        print(f"MinIO connection error: {e}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": f"MinIO connection failed: {str(e)}"},
        )

    uploaded = []
    failed = []

    for file, rel_path in zip(files, paths):
        try:
            content = await file.read()
            rel_path = rel_path.strip().lstrip("/")

            minio_client.put_object(
                bucket_name,
                rel_path,
                io.BytesIO(content),
                length=len(content),
                content_type=file.content_type or "application/octet-stream",
            )
            uploaded.append(rel_path)
            print(f"Successfully uploaded: {rel_path}")

        except Exception as e:
            print(f"Failed to upload {file.filename}: {e}")
            failed.append({"file": file.filename, "error": str(e)})

    return {
        "status": "partial_success" if failed else "success",
        "uploaded": uploaded,
        "failed": failed,
    }
