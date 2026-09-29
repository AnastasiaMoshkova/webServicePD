"""Автономная точка входа в gait API для локальной отладки.
Рабочий сайт запускается из app.main:app.
"""
from fastapi import FastAPI

from app.modalities.gait.router import init_gait_service, router

app = FastAPI(title="PD Gait Analysis API")
init_gait_service()
app.include_router(router)
