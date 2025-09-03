from fastapi import APIRouter, Request
from core.templates import templates

router = APIRouter()


@router.get("/processing")
def catalog(request: Request):
    return templates.TemplateResponse("processing.html", {"request": request})
