from pydantic import BaseModel, Field
from typing import List, Dict, Optional

class GaitSignalInput(BaseModel):
    patient_id: str = Field(..., description="Идентификатор пациента")
    time_series: List[List[float]] = Field(..., description="Массив признаков/временных рядов [T, F]")
    extracted_features: Optional[Dict[str, float]] = Field(default=None, description="Готовые биомаркеры шага")

class GaitPredictionResult(BaseModel):
    patient_id: str
    pd_probability: float = Field(..., ge=0.0, le=1.0, description="Вероятность наличия болезни Паркинсона")
    is_pd_detected: bool = Field(..., description="Флаг детекции")
    freeze_index: Optional[float] = None
    cadence: Optional[float] = None
    status: str = "success"