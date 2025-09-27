"""
BNPL Risk Assessment API Endpoints

Provides HTTP endpoints for real-time BNPL default risk prediction using
the production ML pipeline: feature engineering → multi-model prediction → response.
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field, validator
from typing import Dict, Optional, List
import time
import logging
from datetime import datetime

from flit_ml.features.bnpl_feature_engineering import BNPLFeatureEngineer
from flit_ml.models.bnpl.predictor import BNPLPredictor


# Router for BNPL endpoints
router = APIRouter(prefix="/v1/bnpl", tags=["BNPL Risk Assessment"])

# Global instances (initialized once for performance)
_feature_engineer: Optional[BNPLFeatureEngineer] = None
_predictor: Optional[BNPLPredictor] = None

# Logger for API operations
logger = logging.getLogger(__name__)


# Request/Response Models
class TransactionInput(BaseModel):
    """Input model for BNPL transaction risk assessment."""

    # Transaction details
    transaction_id: str = Field(..., description="Unique transaction identifier")
    customer_id: str = Field(..., description="Customer identifier")
    amount: float = Field(..., gt=0, description="Transaction amount in USD")
    transaction_timestamp: str = Field(..., description="ISO timestamp of transaction")

    # Customer attributes
    customer_credit_score_range: str = Field(..., description="Credit score range: poor|fair|good|excellent")
    customer_age_bracket: str = Field(..., description="Age bracket: 18-24|25-34|35-44|45-54|55+")
    customer_income_bracket: str = Field(..., description="Income bracket: <25k|25k-50k|50k-75k|75k-100k|100k+")
    customer_verification_level: str = Field(..., description="Verification level: unverified|partial|verified")
    customer_tenure_days: int = Field(..., ge=0, description="Days since customer registration")

    # Device context
    device_type: str = Field(..., description="Device type: desktop|mobile|tablet")
    device_is_trusted: bool = Field(..., description="Whether device is trusted")

    # Product details
    product_category: str = Field(..., description="Product category: clothing|electronics|home|sports|beauty")
    product_risk_category: str = Field(..., description="Product risk: low|medium|high")

    # Risk assessment
    risk_score: float = Field(..., ge=0, le=1, description="Current risk score [0,1]")
    risk_level: str = Field(..., description="Risk level: low|medium|high")
    risk_scenario: str = Field(..., description="Risk scenario: high_risk_behavior|impulse_purchase|low_risk_purchase|repeat_customer")

    # Payment details
    payment_provider: str = Field(..., description="BNPL provider: afterpay|klarna|sezzle|zip")
    installment_count: int = Field(..., gt=0, description="Number of installments")
    payment_credit_limit: float = Field(..., gt=0, description="Credit limit for payment method")
    price_comparison_time: float = Field(..., ge=0, description="Time spent comparing prices (seconds)")
    purchase_context: str = Field(..., description="Purchase context: normal|rushed|sale")

    @validator('transaction_timestamp')
    def validate_timestamp(cls, v):
        """Validate ISO timestamp format."""
        try:
            datetime.fromisoformat(v.replace('Z', '+00:00'))
            return v
        except ValueError:
            raise ValueError("transaction_timestamp must be valid ISO format")

    @validator('customer_credit_score_range')
    def validate_credit_score(cls, v):
        if v not in ['poor', 'fair', 'good', 'excellent']:
            raise ValueError("Invalid credit score range")
        return v

    @validator('risk_level')
    def validate_risk_level(cls, v):
        if v not in ['low', 'medium', 'high']:
            raise ValueError("Invalid risk level")
        return v


class RiskAssessmentResponse(BaseModel):
    """Response model for risk assessment."""

    # Request context
    transaction_id: str
    assessment_timestamp: str

    # Model predictions
    predictions: Dict = Field(..., description="Model predictions by name")
    champion_model: str = Field(..., description="Current champion model")

    # Risk classification
    risk_level: str = Field(..., description="Predicted risk level: LOW|MEDIUM|HIGH")
    default_probability: float = Field(..., description="Probability of default [0,1]")

    # Performance metrics
    processing_time_ms: float = Field(..., description="Total processing time")
    model_inference_time_ms: float = Field(..., description="Model inference time")

    # Deployment info
    deployment_mode: str = Field(..., description="Current deployment mode")
    model_version: str = Field(..., description="Model version used")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: str
    model_status: Dict[str, str]
    version: str


# Dependency injection for components
async def get_feature_engineer() -> BNPLFeatureEngineer:
    """Get feature engineer instance (singleton)."""
    global _feature_engineer
    if _feature_engineer is None:
        _feature_engineer = BNPLFeatureEngineer(client=None, verbose=False)
        logger.info("Feature engineer initialized")
    return _feature_engineer


async def get_predictor() -> BNPLPredictor:
    """Get predictor instance (singleton)."""
    global _predictor
    if _predictor is None:
        # Default to shadow mode for comprehensive model comparison
        _predictor = BNPLPredictor(mode="shadow", verbose=False)
        logger.info("Predictor initialized in shadow mode")
    return _predictor


# API Endpoints
@router.post("/risk-assessment", response_model=RiskAssessmentResponse)
async def assess_risk(
    transaction: TransactionInput,
    feature_engineer: BNPLFeatureEngineer = Depends(get_feature_engineer),
    predictor: BNPLPredictor = Depends(get_predictor)
) -> RiskAssessmentResponse:
    """
    Assess default risk for a BNPL transaction.

    Processes the transaction through the complete ML pipeline:
    1. Feature engineering (JSON → 36 features)
    2. Multi-model prediction (4 models in shadow mode)
    3. Risk classification and response formatting

    Returns comprehensive risk assessment with model predictions.
    """
    start_time = time.time()

    try:
        # Step 1: Feature Engineering
        features = feature_engineer.engineer_single_transaction(transaction.dict())

        # Step 2: Multi-Model Prediction
        predictions = predictor.predict(features)

        # Step 3: Risk Classification
        # Use champion model prediction for risk level
        champion_model = predictions.get("champion", "ridge")
        default_prob = predictions.get(champion_model, predictions.get("prediction", 0.0))

        # Classify risk level based on probability thresholds
        if default_prob >= 0.7:
            risk_level = "HIGH"
        elif default_prob >= 0.4:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        # Step 4: Format Response
        processing_time = (time.time() - start_time) * 1000

        return RiskAssessmentResponse(
            transaction_id=transaction.transaction_id,
            assessment_timestamp=datetime.utcnow().isoformat() + "Z",
            predictions=predictions,
            champion_model=champion_model,
            risk_level=risk_level,
            default_probability=default_prob,
            processing_time_ms=round(processing_time, 2),
            model_inference_time_ms=predictions.get("inference_time_ms", 0.0),
            deployment_mode=predictor.mode,
            model_version=predictor.model_version
        )

    except Exception as e:
        logger.error(f"Risk assessment failed for transaction {transaction.transaction_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Risk assessment failed: {str(e)}"
        )


@router.get("/health", response_model=HealthResponse)
async def health_check(
    feature_engineer: BNPLFeatureEngineer = Depends(get_feature_engineer),
    predictor: BNPLPredictor = Depends(get_predictor)
) -> HealthResponse:
    """
    Health check endpoint for monitoring.

    Validates that all ML components are properly loaded and functional.
    """
    try:
        # Check model loading status
        model_info = predictor.get_model_info()

        model_status = {
            "feature_engineer": "healthy",
            "predictor": "healthy",
            "models_loaded": str(model_info["models_loaded"]),
            "champion": model_info["champion"]
        }

        return HealthResponse(
            status="healthy",
            timestamp=datetime.utcnow().isoformat() + "Z",
            model_status=model_status,
            version=predictor.model_version
        )

    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(
            status_code=503,
            detail=f"Service unhealthy: {str(e)}"
        )


@router.get("/models/info")
async def get_model_info(
    predictor: BNPLPredictor = Depends(get_predictor)
) -> Dict:
    """
    Get detailed information about loaded models.

    Returns model configuration, deployment mode, and performance metadata.
    """
    try:
        model_info = predictor.get_model_info()

        # Add metadata from predictor
        if predictor.metadata:
            # Build model performance info for loaded models only
            model_performance = {}
            for name in model_info["models_loaded"]:
                if name in predictor.metadata["models"]:
                    model_data = predictor.metadata["models"][name]
                    if isinstance(model_data, dict) and "auc" in model_data:
                        model_performance[name] = {
                            "auc": model_data["auc"],
                            "discrimination_ratio": model_data.get("discrimination_ratio", "N/A")
                        }

            model_info.update({
                "model_performance": model_performance,
                "deployment_strategy": predictor.metadata.get("deployment_strategy", {}),
                "feature_count": predictor.metadata.get("shared_artifacts", {}).get("total_features", 36)
            })

        return model_info

    except Exception as e:
        logger.error(f"Model info retrieval failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve model info: {str(e)}"
        )