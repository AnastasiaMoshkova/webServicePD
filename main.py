from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from routers import home, feature_extraction, hand_tracking, files_processing

app = FastAPI(limit_max_body_size=3 * 1024 * 1024 * 1024)  # 1 ГБ

app.mount("/static", StaticFiles(directory="static"), name="static")

# Подключаем роуты
app.include_router(home.router)
app.include_router(feature_extraction.router)
app.include_router(hand_tracking.router)
app.include_router(files_processing.router)
