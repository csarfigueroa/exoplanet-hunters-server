from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from app.models.neural_network import NeuralNetworkService

router = APIRouter(prefix="/neural-network", tags=["Neural Network"])

nn_service = NeuralNetworkService()

class TrainRequest(BaseModel):
    epochs: Optional[int] = 50
    learning_rate: Optional[float] = 0.001

class EvaluateRequest(BaseModel):
    input_data: Optional[List[List[float]]] = None

@router.post("/train")
async def train_model(request: TrainRequest):
    try:
        result = nn_service.train_model(
            epochs=request.epochs,
            learning_rate=request.learning_rate
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

@router.post("/evaluate")
async def evaluate_model(request: EvaluateRequest):
    try:
        result = nn_service.evaluate_model(input_data=request.input_data)
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")

@router.get("/status")
async def get_model_status():
    return {
        "model_trained": nn_service.trained,
        "model_exists": nn_service.model is not None
    }