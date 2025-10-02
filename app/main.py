from fastapi import FastAPI
from app.routers import neural_network

app = FastAPI(
    title="Exoplanet Hunter API",
    description="A FastAPI server for classifying exoplanets using neural networks and historical TOI data",
    version="2.0.0"
)

app.include_router(neural_network.router)

@app.get("/")
async def root():
    return {
        "message": "Hello World! Exoplanet Hunter API is running",
        "objective": "Binary classification: Exoplanet vs Non-Exoplanet using TESS Objects of Interest data",
        "endpoints": {
            "train": "POST /exoplanet/train - Train the exoplanet classification model",
            "evaluate": "POST /exoplanet/evaluate - Evaluate model with accuracy percentage",
            "predict": "POST /exoplanet/predict - Predict if a stellar object is an exoplanet",
            "status": "GET /exoplanet/status - Get model status and accuracy information",
            "data_info": "GET /exoplanet/data-info - Get dataset information"
        },
        "data_source": "NASA Exoplanet Archive - TESS Objects of Interest (TOI)",
        "accuracy_display": "All endpoints return accuracy as percentage values"
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "exoplanet-classification"}