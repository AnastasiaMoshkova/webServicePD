from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from core.templates import templates

router = APIRouter()


@router.get("/mimic")
def home(request: Request):
    return templates.TemplateResponse("mimic.html", {"request": request})
