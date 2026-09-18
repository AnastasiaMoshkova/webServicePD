"""Standalone gait API entry point for local debugging.

The production website starts from app.main:app. This module remains usable for
isolated gait API checks without duplicating model-loading logic.
"""
from fastapi import FastAPI

from app.modalities.gait.router import init_gait_service, router

app = FastAPI(title="PD Gait Analysis API")
init_gait_service()
app.include_router(router)
