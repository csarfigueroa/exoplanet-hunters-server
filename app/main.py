from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import neural_network

app = FastAPI(
    title="Exoplanet Hunter API",
    description="A FastAPI server for classifying exoplanets using neural networks and historical TOI data",
    version="2.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

app.include_router(neural_network.router)

@app.get("/")
async def root():
    return {
        "message": "Hello World! Exoplanet Hunter API is running",
        "objective": "Binary classification: Exoplanet vs Non-Exoplanet using TESS Objects of Interest data",
        "endpoints": {
            "classification": {
                "train": "POST /exoplanet/train - Train the exoplanet classification model",
                "evaluate": "POST /exoplanet/evaluate - Evaluate model with accuracy percentage",
                "predict": "POST /exoplanet/predict - Predict if a stellar object is an exoplanet",
                "status": "GET /exoplanet/status - Get model status and accuracy information",
                "data_info": "GET /exoplanet/data-info - Get dataset information"
            }
        },
        "data_source": "NASA Exoplanet Archive - TESS Objects of Interest (TOI)",
        "workflow": "Raw lightcurve → Feature extraction → Classification prediction"
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "exoplanet-classification"}