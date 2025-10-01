from fastapi import FastAPI
from app.routers import neural_network

app = FastAPI(
    title="Neural Network API",
    description="A FastAPI server for training and evaluating neural networks",
    version="1.0.0"
)

app.include_router(neural_network.router)

@app.get("/")
async def root():
    return {
        "message": "Hello World! Neural Network API is running",
        "endpoints": {
            "train": "POST /neural-network/train",
            "evaluate": "POST /neural-network/evaluate",
            "status": "GET /neural-network/status"
        }
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy"}