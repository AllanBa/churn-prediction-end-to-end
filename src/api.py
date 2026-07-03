import time
import logging
import joblib
import torch
import pandas as pd
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel, Field, ConfigDict

# Import our neural network architecture
from src.train_nn import ChurnMLP

# 1. Structured Logging Setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("churn_api")

# 2. FastAPI Application Initialization
app = FastAPI(
    title="Churn Prediction API",
    description="Real-time inference API for predicting telecommunications customer churn using a PyTorch MLP.",
    version="1.0.0"
)

# 3. Load Production Artifacts (Global State)
try:
    preprocessor = joblib.load("models/preprocessor.pkl")
    model_data = torch.load("models/churn_mlp.pth")
    
    model = ChurnMLP(input_dim=model_data["input_dim"])
    model.load_state_dict(model_data["state_dict"])
    model.eval() # Set to evaluation mode
    logger.info("Real Machine Learning model and preprocessor loaded successfully.")
except Exception as e:
    logger.error(f"CRITICAL: Failed to load ML artifacts: {e}")
    model, preprocessor = None, None

# 4. Latency Middleware
@app.middleware("http")
async def log_requests_and_latency(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = (time.time() - start_time) * 1000
    response.headers["X-Process-Time"] = str(process_time)
    logger.info(f"Method: {request.method} | Path: {request.url.path} | Status: {response.status_code} | Latency: {process_time:.2f} ms")
    return response

# 5. Pydantic Schemas (V2)
class CustomerData(BaseModel):
    tenure_months: int = Field(..., alias="Tenure Months", ge=0)
    monthly_charges: float = Field(..., alias="Monthly Charges", ge=0.0)
    total_charges: float = Field(..., alias="Total Charges", ge=0.0)
    cltv: float = Field(..., alias="CLTV", ge=0.0)
    churn_score: float = Field(..., alias="Churn Score", ge=0.0, le=100.0)
    gender: str = Field(..., pattern="^(Male|Female)$")
    senior_citizen: str = Field(..., pattern="^(Yes|No)$")
    partner: str = Field(..., pattern="^(Yes|No)$")
    dependents: str = Field(..., pattern="^(Yes|No)$")
    internet_service: str = Field(..., alias="Internet Service")
    contract: str = Field(...)

    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "Tenure Months": 12,
                "Monthly Charges": 75.5,
                "Total Charges": 906.0,
                "CLTV": 4500,
                "Churn Score": 85,
                "gender": "Female",
                "senior_citizen": "No",
                "partner": "Yes",
                "dependents": "No",
                "Internet Service": "Fiber optic",
                "contract": "Month-to-month"
            }
        }
    )

class PredictionResponse(BaseModel):
    churn_probability: float
    churn_prediction: int
    latency_ms: float

# 6. API Endpoints
@app.get("/health", tags=["System"])
async def health_check():
    if model is None or preprocessor is None:
        raise HTTPException(status_code=503, detail="Model artifacts not loaded.")
    return {"status": "healthy", "message": "API and Real PyTorch Model are active."}

@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict_churn(customer: CustomerData):
    start_time = time.time()
    
    if model is None or preprocessor is None:
        raise HTTPException(status_code=503, detail="Model is unavailable.")
        
    try:
        # Convert JSON to DataFrame
        input_df = pd.DataFrame([customer.model_dump(by_alias=True)])
        
        # Apply strict preprocessing (Scaling & Encoding)
        X_processed = preprocessor.transform(input_df)
        
        # Convert to PyTorch Tensor
        X_tensor = torch.FloatTensor(X_processed)
        
        # Real Inference Forward Pass
        with torch.no_grad():
            probability = model(X_tensor).item()
            
        # Business Threshold logic
        prediction = 1 if probability >= 0.5 else 0
        
        latency = (time.time() - start_time) * 1000
        
        return PredictionResponse(
            churn_probability=round(probability, 4),
            churn_prediction=prediction,
            latency_ms=round(latency, 2)
        )
        
    except Exception as e:
        logger.error(f"Inference failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal ML execution error.")