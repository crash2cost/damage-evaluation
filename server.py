#!/usr/bin/env python3

import argparse
import io
import time
from typing import Dict, List, Optional

import uvicorn
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

from pipeline import (
    Crash2CostPipeline,
    DamageAssessment,
    VehicleAssessment,
    FALLBACK_DAMAGE_TO_PART as DAMAGE_TO_PART,
)

API_TITLE = "Crash2Cost API"
API_DESCRIPTION = "Car damage detection and cost estimation API"
API_VERSION = "1.0.0"

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8000

DEFAULT_CONFIDENCE_THRESHOLD = 0.02
MIN_CONFIDENCE_THRESHOLD = 0.01
MAX_CONFIDENCE_THRESHOLD = 0.9

DEFAULT_ASSUMED_SEVERITY = 2
DEFAULT_ASSUMED_CONFIDENCE = 0.15
DEFAULT_ASSUMED_DAMAGE_TYPE = "suspected damage (low confidence)"
DEFAULT_ASSUMED_PART = "Unknown Area"
DEFAULT_ASSUMED_COST = 1500
DEFAULT_UNKNOWN_PART = "Front Bumper"

HTTP_BAD_REQUEST = 400
HTTP_INTERNAL_ERROR = 500
HTTP_SERVICE_UNAVAILABLE = 503

MS_PER_SECOND = 1000

DEFAULT_CURRENCY = "ILS"

app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline: Optional[Crash2CostPipeline] = None


class DamageResponse(BaseModel):
    damage_type: str
    confidence: float
    severity: int
    action: str
    estimated_cost: float
    bbox: List[float]


class AssessmentResponse(BaseModel):
    success: bool
    damages: List[DamageResponse]
    total_cost: float
    inference_time_ms: float
    car_segment: str
    message: str


class HealthResponse(BaseModel):
    status: str
    models_loaded: Dict[str, bool]


@app.on_event("startup")
async def startup_event() -> None:
    global pipeline
    print("\nLoading Crash2Cost models...")
    try:
        pipeline = Crash2CostPipeline()
        print("Models loaded successfully!\n")
    except (FileNotFoundError, RuntimeError) as e:
        print(f"Failed to load models: {e}")
        raise


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    return HealthResponse(
        status="healthy" if pipeline else "unhealthy",
        models_loaded={
            "detection": pipeline.detection_model is not None if pipeline else False,
            "severity": pipeline.severity_model is not None if pipeline else False,
            "cost": pipeline.cost_model is not None if pipeline else False,
        }
    )


@app.post("/assess")
async def assess_damage(
    file: UploadFile = File(..., description="Car damage image (JPG/PNG)", alias="file"),
    car_segment: str = Query(
        default="Family",
        description="Car segment for cost estimation",
        enum=["Micro", "Family", "Executive", "Luxury", "SUV"]
    ),
    conf_threshold: float = Query(
        default=DEFAULT_CONFIDENCE_THRESHOLD,
        ge=MIN_CONFIDENCE_THRESHOLD,
        le=MAX_CONFIDENCE_THRESHOLD,
        description="Detection confidence threshold"
    ),
    use_tta: bool = Query(
        default=False,
        description="Enable test-time augmentation for higher accuracy (slower)"
    ),
) -> JSONResponse:
    if not pipeline:
        raise HTTPException(
            status_code=HTTP_SERVICE_UNAVAILABLE,
            detail="Models not loaded"
        )

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=HTTP_BAD_REQUEST,
            detail="File must be an image"
        )

    try:
        contents = await file.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")

        start_time = time.time()
        assessment: VehicleAssessment = pipeline.assess_damage(
            pil_image,
            car_segment=car_segment,
            conf_threshold=conf_threshold,
            use_tta=use_tta,
        )
        inference_time = (time.time() - start_time) * MS_PER_SECOND

        if assessment.damages:
            # Return primary damage for backward compatibility
            d = assessment.damages[0]
            part_name = DAMAGE_TO_PART.get(d.damage_type, DEFAULT_UNKNOWN_PART)

            return JSONResponse(content={
                "damageType": d.damage_type,
                "confidence": d.damage_confidence,
                "part": part_name,
                "severity": d.severity,
                "carSegment": car_segment,
                "estimatedCost": int(assessment.total_cost),
                "currency": DEFAULT_CURRENCY,
                "damages": [
                    {
                        "damageType": dmg.damage_type,
                        "confidence": round(dmg.damage_confidence, 4),
                        "part": DAMAGE_TO_PART.get(dmg.damage_type, DEFAULT_UNKNOWN_PART),
                        "severity": dmg.severity,
                        "estimatedCost": int(dmg.estimated_cost),
                        "action": dmg.repair_or_replace,
                        "bbox": list(dmg.detection.bbox),
                    }
                    for dmg in assessment.damages
                ],
                "inferenceTimeMs": round(inference_time, 1),
            })
        else:
            assumed_cost = (
                pipeline.estimate_cost("dent", DEFAULT_ASSUMED_SEVERITY, car_segment)
                if pipeline else DEFAULT_ASSUMED_COST
            )
            return JSONResponse(content={
                "damageType": DEFAULT_ASSUMED_DAMAGE_TYPE,
                "confidence": DEFAULT_ASSUMED_CONFIDENCE,
                "part": DEFAULT_ASSUMED_PART,
                "severity": DEFAULT_ASSUMED_SEVERITY,
                "carSegment": car_segment,
                "estimatedCost": int(assumed_cost),
                "currency": DEFAULT_CURRENCY,
                "damages": [],
                "inferenceTimeMs": round(inference_time, 1),
            })

    except UnidentifiedImageError as e:
        raise HTTPException(
            status_code=HTTP_BAD_REQUEST,
            detail=f"Invalid image format: {str(e)}"
        )
    except ValueError as e:
        raise HTTPException(
            status_code=HTTP_BAD_REQUEST,
            detail=f"Invalid input: {str(e)}"
        )
    except (IOError, RuntimeError) as e:
        raise HTTPException(
            status_code=HTTP_INTERNAL_ERROR,
            detail=f"Processing error: {str(e)}"
        )


@app.get("/")
async def root() -> Dict[str, object]:
    return {
        "name": API_TITLE,
        "version": API_VERSION,
        "docs": "/docs",
        "endpoints": {
            "POST /assess": "Upload image for damage assessment",
            "GET /health": "Health check",
        }
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Crash2Cost API Server")
    parser.add_argument("--host", default=DEFAULT_HOST, help="Host to bind to")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    args = parser.parse_args()

    print(f"""
===========================================================
               Crash2Cost API Server
===========================================================
  Docs:   http://{args.host}:{args.port}/docs
  Health: http://{args.host}:{args.port}/health
===========================================================
""")

    uvicorn.run(
        "server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
