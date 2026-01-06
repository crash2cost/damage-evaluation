from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from io import BytesIO

from crash2cost import (
    DamageClassifier,
    CostEstimator,
    DAMAGE_TO_PART,
    CLASSIFIER_PATH,
    COST_MODEL_PATH,
    PART_ENCODER_PATH,
    SEGMENT_ENCODER_PATH,
)

app = FastAPI(title="Crash2Cost ML Service", version="0.1.0")

MODEL_DEVICE = "cpu"

CLASSIFIER = DamageClassifier(CLASSIFIER_PATH, device=MODEL_DEVICE)
COST_ESTIMATOR = CostEstimator(COST_MODEL_PATH, PART_ENCODER_PATH, SEGMENT_ENCODER_PATH)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/assess")
async def assess_damage(
    file: UploadFile = File(...),
    severity: int = Form(3),
    carSegment: str = Form("Family"),
) -> JSONResponse:
    if severity < 1 or severity > 5:
        raise HTTPException(status_code=400, detail="Severity must be between 1 and 5")

    try:
        raw_bytes = await file.read()
        image = Image.open(BytesIO(raw_bytes)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid image file") from exc

    damage_type, confidence = CLASSIFIER.predict_image(image)
    part_name = DAMAGE_TO_PART.get(damage_type, "Front Bumper")
    estimated_cost = COST_ESTIMATOR.estimate(part_name, severity, carSegment)

    if estimated_cost is None:
        raise HTTPException(status_code=422, detail="Unable to estimate cost for this input")

    payload = {
        "damageType": damage_type,
        "confidence": confidence,
        "part": part_name,
        "severity": severity,
        "carSegment": carSegment,
        "estimatedCost": estimated_cost,
        "currency": "ILS",
    }
    return JSONResponse(content=payload)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8004)
