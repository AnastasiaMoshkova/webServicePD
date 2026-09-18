"""Optional JSON API for gait. The website itself is wired through app.routers.gait."""
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException

from .schemas import GaitPredictionResult, GaitSignalInput
from .service import GaitService

router = APIRouter(prefix="/gait", tags=["Gait Analysis"])
gait_service: Optional[GaitService] = None


def init_gait_service() -> GaitService:
    global gait_service
    if gait_service is None:
        gait_service = GaitService()
    return gait_service


def get_gait_service() -> GaitService:
    service = init_gait_service()
    if not service.ready:
        raise HTTPException(status_code=503, detail=service.models_status())
    return service


@router.post("/predict", response_model=GaitPredictionResult)
async def analyze_gait(data: GaitSignalInput, service: GaitService = Depends(get_gait_service)):
    # This compatibility endpoint accepts already extracted classifier features.
    if not data.extracted_features:
        raise HTTPException(
            status_code=400,
            detail="Для JSON /gait/predict передайте extracted_features. Для сырых CSV используйте /api/process-signal.",
        )
    try:
        from app.modalities.gait.processing.ml_pipeline import classify_features

        pred = classify_features(data.extracted_features, service.classifier, service.classifier_scaler)
        # classify_features returns confidence for the predicted class; expose true PD probability here.
        names = [
            'cadence', 'sample_entropy', 'freeze_index_mean', 'step_time_cv',
            'X_std', 'X_rms', 'X_jerk_std'
        ]
        vec = [[float(data.extracted_features.get(name, 0.0) or 0.0) for name in names]]
        scaled = service.classifier_scaler.transform(vec)
        proba_pd = float(service.classifier.predict_proba(scaled)[0][1])
        return GaitPredictionResult(
            patient_id=data.patient_id,
            pd_probability=proba_pd,
            is_pd_detected=bool(pred["prediction"] == 1),
            freeze_index=data.extracted_features.get("freeze_index_mean"),
            cadence=data.extracted_features.get("cadence"),
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
