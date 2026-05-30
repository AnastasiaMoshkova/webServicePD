from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import JSONResponse
from core.templates import templates

router = APIRouter()


@router.get("/gait")
def home(request: Request):
    return templates.TemplateResponse("gait.html", {"request": request})
