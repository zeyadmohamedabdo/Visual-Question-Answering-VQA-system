"""
FastAPI Backend for VQA System
===============================
This module implements the REST API for serving VQA predictions.

Endpoints:
---------
POST /predict: Submit image and question, get answer
GET /health: Health check
GET /model-info: Get model information

The API accepts images as file uploads and returns JSON responses.

Usage:
------
uvicorn api.main:app --reload --port 8000

Then access:
- API: http://localhost:8000
- Docs: http://localhost:8000/docs
"""

import os
import sys
from pathlib import Path
from typing import Optional, List
from io import BytesIO

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from api.inference import VQAInference, get_inference_engine


# =============================================================================
# API Models (Pydantic)
# =============================================================================

class PredictionResponse(BaseModel):
    """Response model for predictions."""
    question: str
    top_answer: str
    confidence: float
    answers: List[dict]
    success: bool = True
    error: Optional[str] = None


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    model_loaded: bool


class ModelInfoResponse(BaseModel):
    """Response model for model information."""
    device: str
    vocab_size: int
    num_answers: int
    total_parameters: int


# =============================================================================
# FastAPI App
# =============================================================================

app = FastAPI(
    title="VQA API",
    description="Visual Question Answering API - Ask questions about images",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global inference engine (initialized on first request)
inference_engine: Optional[VQAInference] = None


def get_engine() -> VQAInference:
    """Get or create inference engine."""
    global inference_engine
    if inference_engine is None:
        inference_engine = VQAInference()
        inference_engine.load()
    return inference_engine


# =============================================================================
# Endpoints
# =============================================================================

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "VQA API",
        "version": "1.0.0",
        "description": "Visual Question Answering System",
        "endpoints": {
            "predict": "POST /predict - Submit image and question",
            "health": "GET /health - Health check",
            "model-info": "GET /model-info - Model information",
            "docs": "GET /docs - API documentation"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Status"])
async def health_check():
    """
    Health check endpoint.
    
    Returns API status and whether model is loaded.
    """
    global inference_engine
    return HealthResponse(
        status="healthy",
        model_loaded=inference_engine is not None and inference_engine._is_loaded
    )


@app.get("/model-info", response_model=ModelInfoResponse, tags=["Status"])
async def model_info():
    """
    Get model information.
    
    Returns model configuration and parameter counts.
    """
    try:
        engine = get_engine()
        info = engine.get_model_info()
        
        return ModelInfoResponse(
            device=str(info['device']),
            vocab_size=info['vocab_size'],
            num_answers=info['num_answers'],
            total_parameters=info['parameters']['total']
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(
    image: UploadFile = File(..., description="Image file (JPEG, PNG)"),
    question: str = Form(..., description="Question about the image"),
    top_k: int = Form(5, description="Number of top predictions to return")
):
    """
    Predict answer for image-question pair.
    
    Submit an image and a question to get the predicted answer.
    
    Parameters:
    - **image**: Image file (JPEG, PNG, etc.)
    - **question**: Question about the image
    - **top_k**: Number of top predictions (default: 5)
    
    Returns:
    - **question**: Original question
    - **top_answer**: Best predicted answer
    - **confidence**: Confidence score (0-1)
    - **answers**: List of top-k answers with probabilities
    """
    try:
        # Validate image
        if not image.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400, 
                detail="File must be an image (JPEG, PNG, etc.)"
            )
        
        # Validate question
        if not question or len(question.strip()) < 2:
            raise HTTPException(
                status_code=400,
                detail="Question must not be empty"
            )
        
        # Read image bytes
        image_bytes = await image.read()
        
        # Get inference engine and predict
        engine = get_engine()
        result = engine.predict(image_bytes, question.strip(), top_k=top_k)
        
        return PredictionResponse(
            question=result['question'],
            top_answer=result['top_answer'],
            confidence=result['confidence'],
            answers=result['answers'],
            success=True
        )
        
    except HTTPException:
        raise
    except Exception as e:
        return PredictionResponse(
            question=question,
            top_answer="",
            confidence=0.0,
            answers=[],
            success=False,
            error=str(e)
        )


@app.post("/predict-batch", tags=["Prediction"])
async def predict_batch(
    images: List[UploadFile] = File(..., description="Image files"),
    questions: str = Form(..., description="Questions (comma-separated)")
):
    """
    Batch prediction for multiple image-question pairs.
    
    Submit multiple images with corresponding questions.
    Questions should be comma-separated in the same order as images.
    """
    try:
        # Parse questions
        question_list = [q.strip() for q in questions.split(",")]
        
        if len(images) != len(question_list):
            raise HTTPException(
                status_code=400,
                detail=f"Number of images ({len(images)}) must match number of questions ({len(question_list)})"
            )
        
        # Read all image bytes
        image_bytes_list = []
        for img in images:
            if not img.content_type.startswith('image/'):
                raise HTTPException(
                    status_code=400,
                    detail=f"File {img.filename} must be an image"
                )
            image_bytes_list.append(await img.read())
        
        # Get predictions
        engine = get_engine()
        results = engine.predict_batch(image_bytes_list, question_list)
        
        return {
            "success": True,
            "predictions": results
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.on_event("startup")
async def startup_event():
    """Initialize model on startup."""
    print("[API] Starting VQA API server...")
    try:
        # Pre-load the inference engine
        engine = get_engine()
        print(f"[API] Model loaded successfully on {engine.device}")
    except Exception as e:
        print(f"[API] Warning: Could not pre-load model: {e}")
        print("[API] Model will be loaded on first request")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    print("[API] Shutting down VQA API server...")


# =============================================================================
# Main Entry Point
# =============================================================================

def run_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """
    Run the API server.
    
    Args:
        host: Server host
        port: Server port
        reload: Enable auto-reload for development
    """
    uvicorn.run(
        "api.main:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="VQA API Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    args = parser.parse_args()
    
    run_server(host=args.host, port=args.port, reload=args.reload)
