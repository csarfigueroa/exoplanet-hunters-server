from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict
from app.models.neural_network import ExoplanetNeuralNetworkService

router = APIRouter(prefix="/exoplanet", tags=["Exoplanet Classification"])

exoplanet_service = ExoplanetNeuralNetworkService()

class TrainRequest(BaseModel):
    epochs: Optional[int] = 100
    learning_rate: Optional[float] = 0.001
    batch_size: Optional[int] = 64

class EvaluateRequest(BaseModel):
    input_data: Optional[List[List[float]]] = None

class PredictRequest(BaseModel):
    stellar_data: Dict[str, float]

@router.post("/train")
async def train_exoplanet_classifier(request: TrainRequest):
    """
    Train the exoplanet classification model using historical TOI data.
    Returns training accuracy percentage and other metrics.
    """
    try:
        result = exoplanet_service.train_model(
            epochs=request.epochs,
            learning_rate=request.learning_rate,
            batch_size=request.batch_size
        )

        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

@router.post("/evaluate")
async def evaluate_exoplanet_classifier(request: EvaluateRequest):
    """
    Evaluate the trained model and return accuracy percentage with predictions.
    If no input_data provided, uses test samples from the dataset.
    """
    try:
        result = exoplanet_service.evaluate_model(input_data=request.input_data)

        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")

@router.post("/predict")
async def predict_exoplanet(request: PredictRequest):
    """
    Predict if a single stellar object is an exoplanet based on provided data.
    Returns prediction with confidence percentage.
    """
    try:
        if not exoplanet_service.trained:
            raise HTTPException(status_code=400, detail="Model not trained yet. Please train the model first.")

        # Prepare single prediction data
        try:
            processed_data = exoplanet_service.data_processor.prepare_single_prediction(request.stellar_data)
        except Exception as prep_error:
            raise HTTPException(status_code=400, detail=f"Data preparation failed: {str(prep_error)}")

        # Make prediction
        result = exoplanet_service.evaluate_model(input_data=processed_data)

        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])

        # Modify response for single prediction
        if "predictions" in result and len(result["predictions"]) > 0:
            prediction = result["predictions"][0]
            return {
                "message": "Exoplanet classification completed",
                "stellar_object_data": request.stellar_data,
                "classification_result": {
                    "is_exoplanet": prediction["is_exoplanet"],
                    "classification": prediction["classification"],
                    "confidence_level": prediction["confidence_level"],
                    "accuracy_percentage": prediction["accuracy_percentage"],
                    "exoplanet_probability_percentage": prediction["exoplanet_probability_percentage"],
                    "non_exoplanet_probability_percentage": prediction["non_exoplanet_probability_percentage"],
                    "prediction_summary": prediction["prediction_summary"]
                },
                "model_accuracy_percentage": result["model_accuracy_percentage"]
            }
        else:
            return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@router.get("/status")
async def get_model_status():
    """
    Get the current status and information about the trained model.
    """
    try:
        if exoplanet_service.trained:
            return exoplanet_service.get_model_info()
        else:
            return {
                "model_trained": False,
                "model_exists": exoplanet_service.model is not None,
                "message": "Model not trained yet. Use /train endpoint to train the model."
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)}")

@router.get("/data-info")
async def get_data_info():
    """
    Get information about the exoplanet dataset used for training.
    """
    try:
        return exoplanet_service.data_processor.get_data_info()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Data info failed: {str(e)}")