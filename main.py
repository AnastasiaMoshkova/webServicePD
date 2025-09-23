from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from routers import tremor, hand_tracking, mimic, gait, voice

app = FastAPI(limit_max_body_size=3 * 1024 * 1024 * 1024)  # 1 ГБ

app.mount("/static", StaticFiles(directory="static"), name="static")

# Подключаем роуты
app.include_router(tremor.router)
app.include_router(mimic.router)
app.include_router(gait.router)
app.include_router(voice.router)
app.include_router(hand_tracking.router)
