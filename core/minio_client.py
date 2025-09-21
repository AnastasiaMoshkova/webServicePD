import os
import socket
import logging
from minio import Minio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Читаем переменные окружения
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "pd-minio:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "admin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "123admin123")
BUCKET_NAME = os.getenv("MINIO_BUCKET_NAME", "patients")


# def check_minio_connection():
#     """Проверка доступности MinIO (DNS + порт)."""
#     try:
#         ip = socket.gethostbyname(MINIO_ENDPOINT.split(":")[0])
#         logger.info(f"Resolved {MINIO_ENDPOINT} to IP: {ip}")

#         sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#         sock.settimeout(2)
#         result = sock.connect_ex((ip, int(MINIO_ENDPOINT.split(":")[1])))
#         if result == 0:
#             logger.info("MinIO port is open")
#         else:
#             logger.warning("MinIO port is closed")
#         sock.close()

#         return True
#     except Exception as e:
#         logger.error(f"Connection check failed: {e}")
#         return False


def get_minio_client():
    """Создаёт и возвращает клиента MinIO."""
    try:
        client = Minio(
            MINIO_ENDPOINT,
            access_key=MINIO_ACCESS_KEY,
            secret_key=MINIO_SECRET_KEY,
            secure=False,
        )
        logger.info(f"MinIO client initialized with endpoint: {MINIO_ENDPOINT}")
        return client
    except Exception as e:
        logger.error(f"Failed to initialize MinIO client: {e}")
        return None


# Инициализация клиента при импорте модуля
minio_client = get_minio_client()
