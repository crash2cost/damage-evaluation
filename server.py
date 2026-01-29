#!/usr/bin/env python3
"""
Crash2Cost API Server
=====================
FastAPI server for car damage cost estimation.

Usage:
    python server.py                    # Start server on port 8000
    python server.py --port 8080        # Custom port
    
Endpoints:
    POST /assess - Upload image for damage assessment
    GET /health  - Health check
"""

import argparse
import io
import time
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image
import uvicorn

from pipeline import Crash2CostPipeline, VehicleAssessment, DamageAssessment, DAMAGE_TO_PART

# Initialize FastAPI app
app = FastAPI(
    title="Crash2Cost API",
    description="Car damage detection and cost estimation API",
    version="1.0.0",
)

# Enable CORS for web frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline instance (loaded once)
pipeline: Optional[Crash2CostPipeline] = None


class DamageResponse(BaseModel):
    """Single damage item response."""
    damage_type: str
    confidence: float
    severity: int
    action: str
    estimated_cost: float
    bbox: List[float]


class AssessmentResponse(BaseModel):
    """Full assessment response."""
    success: bool
    damages: List[DamageResponse]
    total_cost: float
    inference_time_ms: float
    car_segment: str
    message: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    models_loaded: dict


@app.on_event("startup")
async def startup_event():
    """Load models on server startup."""
    global pipeline
    print("\n🚀 Loading Crash2Cost models...")
    try:
        pipeline = Crash2CostPipeline()
        print("✅ Models loaded successfully!\n")
    except Exception as e:
        print(f"❌ Failed to load models: {e}")
        raise


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check if the server and models are ready."""
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
        default=0.02,
        ge=0.01,
        le=0.9,
        description="Detection confidence threshold"
    ),
):
    """
    Assess car damage from an uploaded image.
    
    Returns detected damages with severity levels and cost estimates.
    """
    if not pipeline:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    # Validate image type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read and convert image
        contents = await file.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Run assessment
        start_time = time.time()
        assessment: VehicleAssessment = pipeline.assess_damage(
            pil_image,
            car_segment=car_segment,
            conf_threshold=conf_threshold,
        )
        inference_time = (time.time() - start_time) * 1000
        
        # Return in format expected by Java backend (PythonAssessmentResponse)
        if assessment.damages:
            # Get first damage for primary response
            d = assessment.damages[0]
            part_name = DAMAGE_TO_PART.get(d.damage_type, "Front Bumper")
            
            return JSONResponse(content={
                "damageType": d.damage_type,
                "confidence": d.damage_confidence,
                "part": part_name,
                "severity": d.severity,
                "carSegment": car_segment,
                "estimatedCost": int(assessment.total_cost),
                "currency": "ILS"
            })
        else:
            # Assume some damage if nothing is detected (low confidence fallback)
            assumed_severity = 2
            assumed_damage_type = "suspected damage (low confidence)"
            assumed_part = "Unknown Area"
            assumed_cost = pipeline.estimate_cost("dent", assumed_severity, car_segment) if pipeline else 1500
            return JSONResponse(content={
                "damageType": assumed_damage_type,
                "confidence": 0.15,
                "part": assumed_part,
                "severity": assumed_severity,
                "carSegment": car_segment,
                "estimatedCost": int(assumed_cost),
                "currency": "ILS"
            })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


@app.get("/")
async def root():
    """API info."""
    return {
        "name": "Crash2Cost API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "POST /assess": "Upload image for damage assessment",
            "GET /health": "Health check",
        }
    }


def main():
    parser = argparse.ArgumentParser(description="Crash2Cost API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    args = parser.parse_args()
    
    print(f"""
╔══════════════════════════════════════════════════════════╗
║               🚗 Crash2Cost API Server                   ║
╠══════════════════════════════════════════════════════════╣
║  Docs:   http://{args.host}:{args.port}/docs                        ║
║  Health: http://{args.host}:{args.port}/health                      ║
╚══════════════════════════════════════════════════════════╝
""")
    
    uvicorn.run(
        "server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
