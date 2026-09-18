import os

from app.routers import gait, hand_tracking, mimic, tremor, voice
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
from app.modalities.gait.service import GaitService
app = FastAPI()

# Перенаправление всех HTTP запросов на HTTPS.
# В проде (за nginx с сертификатом) — обязательно. Для локального запуска
# без TLS (venv/uvicorn --reload или docker без nginx) редирект на
# https://localhost сломает доступ, т.к. сертификата нет — отключается
# переменной окружения HTTPS_REDIRECT=0.
if os.getenv("HTTPS_REDIRECT", "1") != "0":
    app.add_middleware(HTTPSRedirectMiddleware)
# Примечание: `limit_max_body_size` не является параметром FastAPI/Starlette —
# передавался в конструктор молча игнорировался, реального лимита на размер
# тела запроса не было. Если лимит нужен, добавлять через отдельный middleware.

app.mount("/static", StaticFiles(directory="static"), name="static")

MODELS_DIR = os.path.join(os.path.dirname(__file__), "modalities", "gait", "models")
gait_service_instance = GaitService(models_dir=MODELS_DIR)
gait_service_instance.load_artifacts()

# Подключаем роуты
app.include_router(tremor.router)
app.include_router(mimic.router)
app.include_router(gait.router)
app.include_router(voice.router)
app.include_router(hand_tracking.router)
