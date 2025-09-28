# project2-ml-cicd/src/inference/app.py - Windows Compatible
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import joblib
import numpy as np
import logging
from datetime import datetime, timezone
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="ML Model API",
    description="Production ML model serving with health checks",
    version="1.0.0"
)

# Pydantic models
class PredictionRequest(BaseModel):
    features: List[float]

class PredictionResponse(BaseModel):
    prediction: int
    confidence: float
    timestamp: datetime

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    timestamp: datetime

# Global variables
model = None
scaler = None
model_loaded = False

@app.on_event("startup")
async def load_model():
    """Load model and scaler on startup"""
    global model, scaler, model_loaded
    
    try:
        # Look for models in the models directory
        model_path = os.path.join('models', 'model.pkl')
        scaler_path = os.path.join('models', 'scaler.pkl')
        
        # Also check environment variables
        model_path = os.getenv('MODEL_PATH', model_path)
        scaler_path = os.getenv('SCALER_PATH', scaler_path)
        
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            model_loaded = True
            logger.info(f"✅ Model loaded from {model_path}")
            logger.info(f"✅ Scaler loaded from {scaler_path}")
        else:
            logger.warning("⚠️ Model files not found. Available files:")
            for root, dirs, files in os.walk('.'):
                for file in files:
                    if file.endswith('.pkl'):
                        logger.warning(f"  Found: {os.path.join(root, file)}")
            logger.warning("Run training first: python src/training/train.py")
            model_loaded = False
            
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        model_loaded = False

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        model_loaded=model_loaded,
        timestamp=datetime.now(timezone.utc)
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make a prediction"""
    
    if not model_loaded:
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded. Please train a model first by running: python src/training/train.py"
        )
    
    try:
        # Validate input
        if len(request.features) != 20:  # Our sample dataset has 20 features
            raise HTTPException(
                status_code=400,
                detail=f"Expected 20 features, got {len(request.features)}"
            )
        
        # Prepare features
        features = np.array(request.features).reshape(1, -1)
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]
        confidence = float(max(probabilities))
        
        logger.info(f"Prediction made: {prediction} (confidence: {confidence:.3f})")
        
        return PredictionResponse(
            prediction=int(prediction),
            confidence=confidence,
            timestamp=datetime.now(timezone.utc)
        )
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "ML Model API",
        "version": "1.0.0",
        "status": "healthy" if model_loaded else "degraded",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "docs": "/docs",
            "metrics": "/metrics"
        },
        "model_info": {
            "loaded": model_loaded,
            "expected_features": 20,
            "model_type": "RandomForestClassifier"
        }
    }

@app.get("/metrics")
async def get_metrics():
    """Basic metrics endpoint"""
    return {
        "model_loaded": model_loaded,
        "timestamp": datetime.now(timezone.utc),
        "version": "1.0.0",
        "system_info": {
            "python_version": "3.13",
            "platform": "Windows"
        }
    }

@app.get("/model-info")
async def get_model_info():
    """Get detailed model information"""
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Try to load training results
    try:
        with open(os.path.join('models', 'training_result.json'), 'r') as f:
            training_info = json.load(f)
    except:
        training_info = {"error": "Training results not found"}
    
    return {
        "model_type": str(type(model).__name__),
        "feature_count": 20,
        "training_results": training_info,
        "model_loaded": model_loaded
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)