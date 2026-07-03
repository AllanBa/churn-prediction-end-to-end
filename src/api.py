import time
import logging
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel, Field
import pandas as pd

# 1. Structured Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("churn_api")

# 2. FastAPI Application Initialization
app = FastAPI(
    title="Churn Prediction API",
    description="Real-time inference API for predicting telecommunications customer churn.",
    version="1.0.0"
)

# 3. Latency Middleware (Observability)
@app.middleware("http")
async def log_requests_and_latency(request: Request, call_next):
    start_time = time.time()
    
    response = await call_next(request)
    
    process_time = (time.time() - start_time) * 1000 # in milliseconds
    response.headers["X-Process-Time"] = str(process_time)
    
    logger.info(
        f"Method: {request.method} | "
        f"Path: {request.url.path} | "
        f"Status: {response.status_code} | "
        f"Latency: {process_time:.2f} ms"
    )
    return response

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

    class Config:
        populate_by_name = True
        json_schema_extra = {
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

class PredictionResponse(BaseModel):
    churn_probability: float
    churn_prediction: int
    latency_ms: float

@app.get("/health", tags=["System"])
async def health_check():
    """Smoke test endpoint to verify API is running."""
    logger.info("Health check endpoint accessed.")
    return {"status": "healthy", "message": "Churn Prediction API is up and running."}

@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict_churn(customer: CustomerData):
    """Endpoint to predict churn risk for a single customer."""
    start_time = time.time()
    
    try:
        input_data = pd.DataFrame([customer.model_dump(by_alias=True)])
        
        mock_probability = 0.82 if customer.churn_score > 80 else 0.15
        mock_prediction = 1 if mock_probability >= 0.5 else 0
        
        latency = (time.time() - start_time) * 1000
        
        return PredictionResponse(
            churn_probability=mock_probability,
            churn_prediction=mock_prediction,
            latency_ms=latency
        )
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal inference error.")