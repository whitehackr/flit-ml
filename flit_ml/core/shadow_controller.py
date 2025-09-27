"""
BNPL Shadow Mode Controller

Orchestrates experiment management, prediction logging, and business decision integration
for A/B testing and champion/challenger model evaluation in production environments.

This controller separates prediction generation from deployment strategy and business logic,
enabling sophisticated experiment management while maintaining clean separation of concerns.

Architecture reference: docs/architecture/shadow_mode_controller.md
"""

import json
import logging
import time
import uuid
import hashlib
import asyncio
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Protocol
from enum import Enum
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import threading
from collections import deque
import numpy as np

from flit_ml.models.bnpl.predictor import BNPLPredictor
from flit_ml.features.bnpl_feature_engineering import BNPLFeatureEngineer

# Optional MLflow integration
try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


class ExperimentStatus(Enum):
    """Experiment lifecycle states."""
    DRAFT = "draft"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


class DecisionPolicy(Enum):
    """Business decision policies for risk assessment."""
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"
    CUSTOM = "custom"


@dataclass
class ExperimentConfig:
    """Configuration for A/B testing experiments."""
    experiment_id: str
    name: str
    description: str
    traffic_allocation: Dict[str, float]
    models: List[str]
    start_date: datetime
    end_date: Optional[datetime]
    success_metric: str
    min_sample_size: int
    confidence_level: float = 0.95
    status: ExperimentStatus = ExperimentStatus.DRAFT
    created_by: str = "system"
    mlflow_experiment_id: Optional[str] = None

    def __post_init__(self):
        """Validate experiment configuration."""
        total_traffic = sum(self.traffic_allocation.values())
        if abs(total_traffic - 1.0) > 0.01:
            raise ValueError(f"Traffic allocation must sum to 1.0, got {total_traffic}")


@dataclass
class PredictionLog:
    """Structured logging format for model predictions."""
    prediction_id: str
    transaction_id: str
    customer_id: str
    timestamp: datetime

    # Model outputs
    model_predictions: Dict[str, float]
    selected_model: str
    selected_prediction: float

    # Business decision
    business_decision: str
    risk_level: str
    decision_policy: str
    risk_thresholds: Dict[str, float]

    # Experiment context
    experiment_id: Optional[str] = None
    experiment_name: Optional[str] = None
    traffic_segment: str = "champion"

    # Performance tracking
    processing_time_ms: float = 0.0
    features_hash: str = ""
    model_version: str = "v0.1.0"

    # MLflow integration
    mlflow_run_id: Optional[str] = None


class PredictionStorage(Protocol):
    """Storage abstraction for prediction logging."""

    def store_prediction(self, prediction_log: PredictionLog) -> None:
        """Store prediction log with implementation-specific logic."""
        ...

    def get_recent_predictions(self, hours: int) -> List[PredictionLog]:
        """Retrieve recent predictions for analysis."""
        ...

    def get_experiment_data(self, experiment_id: str) -> List[PredictionLog]:
        """Get all predictions for specific experiment."""
        ...


class InMemoryPredictionStorage:
    """In-memory storage implementation for development and testing."""

    def __init__(self, max_size: int = 10000):
        self.predictions: deque = deque(maxlen=max_size)
        self._lock = threading.Lock()

    def store_prediction(self, prediction_log: PredictionLog) -> None:
        """Store prediction in memory with thread safety."""
        with self._lock:
            self.predictions.append(prediction_log)

    def get_recent_predictions(self, hours: int) -> List[PredictionLog]:
        """Get predictions from the last N hours."""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)

        with self._lock:
            return [p for p in self.predictions if p.timestamp >= cutoff_time]

    def get_experiment_data(self, experiment_id: str) -> List[PredictionLog]:
        """Get all predictions for specific experiment."""
        with self._lock:
            return [p for p in self.predictions if p.experiment_id == experiment_id]


class ExperimentManager:
    """Manages A/B testing experiments and traffic allocation."""

    def __init__(self, verbose: bool = True):
        self.experiments: Dict[str, ExperimentConfig] = {}
        self.active_experiment: Optional[str] = None
        self.verbose = verbose
        self.logger = logging.getLogger(__name__)

        # Experiment cache for performance
        self._experiment_cache = {}
        self._cache_ttl = 300  # 5 minutes

    def create_experiment(self, config: ExperimentConfig) -> str:
        """Create new A/B testing experiment."""
        self.experiments[config.experiment_id] = config

        if self.verbose:
            print(f"🧪 Created experiment: {config.name}")
            print(f"   Traffic: {config.traffic_allocation}")
            print(f"   Models: {config.models}")

        return config.experiment_id

    def start_experiment(self, experiment_id: str) -> bool:
        """Activate an experiment."""
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")

        experiment = self.experiments[experiment_id]
        experiment.status = ExperimentStatus.ACTIVE
        self.active_experiment = experiment_id

        # Clear cache to force reload
        self._experiment_cache.clear()

        if self.verbose:
            print(f"🚀 Started experiment: {experiment.name}")

        return True

    def stop_experiment(self, experiment_id: str) -> bool:
        """Deactivate an experiment."""
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")

        experiment = self.experiments[experiment_id]
        experiment.status = ExperimentStatus.COMPLETED

        if self.active_experiment == experiment_id:
            self.active_experiment = None

        self._experiment_cache.clear()

        if self.verbose:
            print(f"🛑 Stopped experiment: {experiment.name}")

        return True

    def get_active_experiment(self) -> Optional[ExperimentConfig]:
        """Get currently active experiment with caching."""
        if not self.active_experiment:
            return None

        # Check cache first
        cache_key = f"active_{self.active_experiment}"
        if cache_key in self._experiment_cache:
            cached_exp, timestamp = self._experiment_cache[cache_key]
            if time.time() - timestamp < self._cache_ttl:
                return cached_exp

        # Get from storage and cache
        experiment = self.experiments.get(self.active_experiment)
        if experiment and experiment.status == ExperimentStatus.ACTIVE:
            self._experiment_cache[cache_key] = (experiment, time.time())
            return experiment

        return None

    def assign_traffic(self, customer_id: str, experiment: ExperimentConfig) -> str:
        """Deterministically assign customer to traffic segment."""
        # Use hash of customer_id for consistent assignment
        hash_input = f"{customer_id}_{experiment.experiment_id}"
        hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16) % 100

        cumulative = 0
        for segment, percentage in experiment.traffic_allocation.items():
            cumulative += percentage * 100
            if hash_value < cumulative:
                return segment

        # Fallback to first segment
        return list(experiment.traffic_allocation.keys())[0]

    def apply_experiment_logic(self, predictions: Dict, transaction_data: Dict) -> Dict:
        """Apply A/B testing logic to select model for business decision."""
        # Default: use champion model
        champion_model = predictions.get("champion", "ridge")
        selected_model = champion_model
        selected_prediction = predictions.get(champion_model, 0.0)

        experiment_info = {
            "selected_model": selected_model,
            "selected_prediction": selected_prediction,
            "experiment_id": None,
            "experiment_name": None,
            "traffic_segment": "champion"
        }

        # Check for active experiment
        experiment = self.get_active_experiment()
        if experiment:
            customer_id = transaction_data.get("customer_id", "")
            traffic_segment = self.assign_traffic(customer_id, experiment)

            # Map traffic segment to model
            if traffic_segment in experiment.traffic_allocation:
                segment_index = list(experiment.traffic_allocation.keys()).index(traffic_segment)
                if segment_index < len(experiment.models):
                    selected_model = experiment.models[segment_index]
                    selected_prediction = predictions.get(selected_model, selected_prediction)

                    experiment_info.update({
                        "selected_model": selected_model,
                        "selected_prediction": selected_prediction,
                        "experiment_id": experiment.experiment_id,
                        "experiment_name": experiment.name,
                        "traffic_segment": traffic_segment
                    })

        return experiment_info


class DecisionManager:
    """Manages business decision policies and risk thresholds."""

    def __init__(self, initial_policy: DecisionPolicy = DecisionPolicy.BALANCED):
        self.current_policy = initial_policy

        # Standard decision policies
        self.decision_policies = {
            DecisionPolicy.CONSERVATIVE: {"high": 0.5, "medium": 0.25},
            DecisionPolicy.BALANCED: {"high": 0.7, "medium": 0.4},
            DecisionPolicy.AGGRESSIVE: {"high": 0.8, "medium": 0.5},
            DecisionPolicy.CUSTOM: {"high": 0.7, "medium": 0.4}  # Default, can be overridden
        }

    def make_business_decision(self, prediction_score: float, model_name: str) -> Tuple[str, str]:
        """Convert model prediction to business decision using current policy."""
        thresholds = self.decision_policies[self.current_policy]

        if prediction_score >= thresholds["high"]:
            return "deny", "HIGH"
        elif prediction_score >= thresholds["medium"]:
            return "manual_review", "MEDIUM"
        else:
            return "approve", "LOW"

    def update_policy(self, policy: DecisionPolicy, custom_thresholds: Optional[Dict[str, float]] = None):
        """Update decision policy and optionally set custom thresholds."""
        if policy == DecisionPolicy.CUSTOM and custom_thresholds:
            self.decision_policies[DecisionPolicy.CUSTOM] = custom_thresholds

        self.current_policy = policy

    def get_current_thresholds(self) -> Dict[str, float]:
        """Get current risk thresholds."""
        return self.decision_policies[self.current_policy].copy()


class ShadowController:
    """
    Orchestrates shadow mode deployment with A/B testing and business integration.

    Separates prediction generation from experiment management and business logic,
    enabling sophisticated deployment strategies while maintaining predictor simplicity.
    """

    def __init__(self,
                 predictor: Optional[BNPLPredictor] = None,
                 feature_engineer: Optional[BNPLFeatureEngineer] = None,
                 storage: Optional[PredictionStorage] = None,
                 verbose: bool = True,
                 enable_mlflow: bool = True):
        """
        Initialize Shadow Mode Controller.

        Args:
            predictor: Multi-model predictor instance
            feature_engineer: Feature engineering instance
            storage: Prediction storage implementation
            verbose: Enable verbose logging
            enable_mlflow: Enable MLflow experiment tracking
        """
        self.predictor = predictor or BNPLPredictor(mode="shadow", verbose=False)
        self.feature_engineer = feature_engineer or BNPLFeatureEngineer(client=None, verbose=False)
        self.storage = storage or InMemoryPredictionStorage()
        self.verbose = verbose
        self.enable_mlflow = enable_mlflow and MLFLOW_AVAILABLE

        # Component managers
        self.experiment_manager = ExperimentManager(verbose=verbose)
        self.decision_manager = DecisionManager()

        # Setup MLflow if available and enabled
        self._setup_mlflow()

        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO if verbose else logging.WARNING)

        if self.verbose:
            print(f"🎯 Shadow Controller initialized")
            print(f"   Predictor mode: {self.predictor.mode}")
            print(f"   Models loaded: {list(self.predictor.models.keys())}")
            print(f"   Storage type: {type(self.storage).__name__}")
            if self.enable_mlflow:
                print(f"   MLflow tracking: enabled")

    def assess_risk_with_logging(self, transaction_data: Dict, log_prediction: bool = True) -> Dict:
        """
        Assess transaction risk with comprehensive logging and experiment management.

        Args:
            transaction_data: Transaction data dict (API input format)
            log_prediction: Whether to log prediction to storage

        Returns:
            Enhanced response with experiment information and business decision
        """
        start_time = time.time()
        prediction_id = str(uuid.uuid4())

        try:
            # Step 1: Feature Engineering
            features = self.feature_engineer.engineer_single_transaction(transaction_data)
            features_hash = self._compute_features_hash(features)

            # Step 2: Model Predictions
            predictions = self.predictor.predict(features)

            # Step 3: Apply Experiment Logic
            experiment_info = self.experiment_manager.apply_experiment_logic(predictions, transaction_data)

            # Step 4: Business Decision
            business_decision, risk_level = self.decision_manager.make_business_decision(
                experiment_info["selected_prediction"],
                experiment_info["selected_model"]
            )

            # Step 5: Async Logging
            processing_time = (time.time() - start_time) * 1000

            if log_prediction:
                prediction_log = PredictionLog(
                    prediction_id=prediction_id,
                    transaction_id=transaction_data.get("transaction_id", "unknown"),
                    customer_id=transaction_data.get("customer_id", "unknown"),
                    timestamp=datetime.utcnow(),
                    model_predictions=predictions,
                    selected_model=experiment_info["selected_model"],
                    selected_prediction=experiment_info["selected_prediction"],
                    business_decision=business_decision,
                    risk_level=risk_level,
                    decision_policy=self.decision_manager.current_policy.value,
                    risk_thresholds=self.decision_manager.get_current_thresholds(),
                    experiment_id=experiment_info.get("experiment_id"),
                    experiment_name=experiment_info.get("experiment_name"),
                    traffic_segment=experiment_info.get("traffic_segment", "champion"),
                    processing_time_ms=processing_time,
                    features_hash=features_hash,
                    model_version=self.predictor.model_version
                )

                # Async logging (non-blocking) - handle event loop gracefully
                try:
                    asyncio.create_task(self._log_prediction_async(prediction_log, experiment_info))
                except RuntimeError:
                    # No event loop running, log synchronously for testing
                    try:
                        self.storage.store_prediction(prediction_log)
                        # Also try MLflow logging for testing
                        if self.enable_mlflow:
                            self._log_to_mlflow(prediction_log, experiment_info)
                    except Exception as e:
                        self.logger.error(f"Sync prediction logging failed: {str(e)}")

            # Step 6: Format Response
            response = {
                "prediction_id": prediction_id,
                "transaction_id": transaction_data.get("transaction_id"),
                "assessment_timestamp": datetime.utcnow().isoformat() + "Z",

                # Business response
                "business_decision": business_decision,
                "risk_level": risk_level,
                "default_probability": experiment_info["selected_prediction"],

                # Model information
                "selected_model": experiment_info["selected_model"],
                "all_predictions": predictions,
                "champion_model": predictions.get("champion", "ridge"),

                # Experiment information
                "experiment_id": experiment_info.get("experiment_id"),
                "experiment_name": experiment_info.get("experiment_name"),
                "traffic_segment": experiment_info.get("traffic_segment"),

                # Performance metrics
                "processing_time_ms": round(processing_time, 2),
                "model_inference_time_ms": predictions.get("inference_time_ms", 0.0),

                # Configuration
                "decision_policy": self.decision_manager.current_policy.value,
                "risk_thresholds": self.decision_manager.get_current_thresholds(),
                "deployment_mode": self.predictor.mode,
                "model_version": self.predictor.model_version
            }

            if self.verbose:
                print(f"🎯 Risk assessment completed in {processing_time:.1f}ms")
                print(f"   Decision: {business_decision} (risk: {risk_level})")
                print(f"   Model: {experiment_info['selected_model']} ({experiment_info['selected_prediction']:.3f})")

            return response

        except Exception as e:
            self.logger.error(f"Risk assessment failed: {str(e)}")
            raise

    async def _log_prediction_async(self, prediction_log: PredictionLog, experiment_info: Optional[Dict] = None):
        """Log prediction asynchronously to avoid blocking API response."""
        try:
            # Store prediction in Redis/storage
            self.storage.store_prediction(prediction_log)

            # Log to MLflow if enabled and experiment info available
            if experiment_info and self.enable_mlflow:
                # Run MLflow logging in a separate thread to avoid blocking
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    executor.submit(self._log_to_mlflow, prediction_log, experiment_info)

        except Exception as e:
            self.logger.error(f"Async prediction logging failed: {str(e)}")
            # Could implement retry logic or fallback storage

    def _compute_features_hash(self, features) -> str:
        """Compute hash of feature values for debugging consistency."""
        try:
            feature_values = features.values.flatten()
            return str(abs(hash(feature_values.tobytes())))[:8]
        except Exception:
            return "unknown"

    def get_recent_performance(self, hours: int = 24) -> Dict:
        """Get recent system performance metrics."""
        try:
            recent_predictions = self.storage.get_recent_predictions(hours)

            if not recent_predictions:
                return {
                    "period": f"Last {hours} hours",
                    "total_predictions": 0,
                    "message": "No predictions in time period"
                }

            # Compute performance metrics
            total_predictions = len(recent_predictions)
            avg_processing_time = np.mean([p.processing_time_ms for p in recent_predictions])

            decisions = [p.business_decision for p in recent_predictions]
            approval_rate = decisions.count("approve") / total_predictions
            manual_review_rate = decisions.count("manual_review") / total_predictions
            denial_rate = decisions.count("deny") / total_predictions

            # Model usage statistics
            models_used = {}
            for pred in recent_predictions:
                model = pred.selected_model
                models_used[model] = models_used.get(model, 0) + 1

            return {
                "period": f"Last {hours} hours",
                "total_predictions": total_predictions,
                "avg_processing_time_ms": round(avg_processing_time, 2),
                "approval_rate": round(approval_rate, 3),
                "manual_review_rate": round(manual_review_rate, 3),
                "denial_rate": round(denial_rate, 3),
                "models_used": models_used,
                "predictions_per_hour": round(total_predictions / hours, 1)
            }

        except Exception as e:
            self.logger.error(f"Performance analysis failed: {str(e)}")
            return {"error": "Performance analysis failed"}

    def get_experiment_analysis(self, experiment_id: str, min_sample_size: int = 100) -> Dict:
        """Analyze experiment performance with basic statistical tests."""
        try:
            experiment_data = self.storage.get_experiment_data(experiment_id)

            if len(experiment_data) < min_sample_size:
                return {
                    "experiment_id": experiment_id,
                    "status": "insufficient_data",
                    "sample_size": len(experiment_data),
                    "min_required": min_sample_size
                }

            # Group by traffic segment
            segments = {}
            for pred in experiment_data:
                segment = pred.traffic_segment
                if segment not in segments:
                    segments[segment] = []
                segments[segment].append(pred)

            # Compute metrics per segment
            segment_metrics = {}
            for segment, predictions in segments.items():
                decisions = [p.business_decision for p in predictions]
                approval_rate = decisions.count("approve") / len(decisions)
                avg_risk_score = np.mean([p.selected_prediction for p in predictions])

                segment_metrics[segment] = {
                    "sample_size": len(predictions),
                    "approval_rate": round(approval_rate, 3),
                    "avg_risk_score": round(avg_risk_score, 3)
                }

            return {
                "experiment_id": experiment_id,
                "status": "sufficient_data",
                "total_sample_size": len(experiment_data),
                "segment_metrics": segment_metrics,
                "analysis_timestamp": datetime.utcnow().isoformat()
            }

        except Exception as e:
            self.logger.error(f"Experiment analysis failed: {str(e)}")
            return {"error": "Experiment analysis failed"}

    def get_controller_info(self) -> Dict:
        """Get comprehensive controller status information."""
        return {
            "status": "operational",
            "predictor_mode": self.predictor.mode,
            "models_loaded": list(self.predictor.models.keys()),
            "current_policy": self.decision_manager.current_policy.value,
            "decision_thresholds": self.decision_manager.get_current_thresholds(),
            "active_experiment": self.experiment_manager.active_experiment,
            "total_experiments": len(self.experiment_manager.experiments),
            "storage_type": type(self.storage).__name__,
            "component_versions": {
                "model_version": self.predictor.model_version,
                "api_version": "v0.1.0"
            },
            "mlflow_enabled": self.enable_mlflow
        }

    def _setup_mlflow(self):
        """Setup MLflow tracking for experiment management."""
        if not self.enable_mlflow:
            return

        try:
            # Configure MLflow for production use
            mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
            mlflow.set_tracking_uri(mlflow_tracking_uri)

            # Set experiment name
            experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "bnpl_shadow_mode")
            mlflow.set_experiment(experiment_name)

            if self.verbose:
                print(f"📊 MLflow configured: {mlflow_tracking_uri}")

        except Exception as e:
            self.logger.warning(f"MLflow setup failed: {e}")
            self.enable_mlflow = False

    def _log_to_mlflow(self, prediction_log: PredictionLog, experiment_info: Dict):
        """Log prediction and experiment data to MLflow."""
        if not self.enable_mlflow:
            return

        try:
            with mlflow.start_run(run_name=f"prediction_{prediction_log.prediction_id[:8]}"):
                # Log prediction metrics
                mlflow.log_metric("default_probability", prediction_log.default_probability)
                mlflow.log_metric("processing_time_ms", prediction_log.processing_time_ms)

                # Log experiment parameters
                mlflow.log_param("model_selected", prediction_log.selected_model)
                mlflow.log_param("business_decision", prediction_log.business_decision)
                mlflow.log_param("risk_level", prediction_log.risk_level)
                mlflow.log_param("decision_policy", prediction_log.decision_policy)
                mlflow.log_param("experiment_id", prediction_log.experiment_id or "none")
                mlflow.log_param("traffic_segment", prediction_log.traffic_segment)

                # Log model predictions as metrics
                for model_name, prediction in experiment_info["all_predictions"].items():
                    if isinstance(prediction, (int, float)):
                        mlflow.log_metric(f"model_{model_name}_prediction", prediction)

                # Log risk thresholds
                for threshold_name, threshold_value in prediction_log.risk_thresholds.items():
                    mlflow.log_param(f"threshold_{threshold_name}", threshold_value)

        except Exception as e:
            self.logger.warning(f"MLflow logging failed: {e}")


# Factory Functions for Production Deployment

def create_production_storage() -> PredictionStorage:
    """
    Create storage implementation based on environment configuration.

    Environment Variables (Railway deployment):
        REDIS_URL: Railway-provided Redis connection URL
        ML_PREDICTION_TTL: TTL for predictions in seconds (default: 2592000 = 30 days)
        ML_STORAGE_VERBOSE: Enable verbose logging (default: false)

    Development (.env.redis file in project root):
        REDIS_URL=redis://localhost:6379/0
        ML_PREDICTION_TTL=2592000
        ML_STORAGE_VERBOSE=true

    Returns:
        Storage implementation appropriate for the environment
    """
    # Load .env.redis file in development if it exists
    from dotenv import load_dotenv

    env_redis_path = os.path.join(os.getcwd(), '.env.redis')
    if os.path.exists(env_redis_path):
        load_dotenv(env_redis_path)

    verbose = os.getenv("ML_STORAGE_VERBOSE", "false").lower() == "true"

    # Check if Redis URL is available (Railway sets this automatically)
    redis_url = os.getenv("REDIS_URL")

    if redis_url:
        try:
            # Import Redis storage (only when needed to avoid dependency issues)
            from flit_ml.core.redis_storage import RedisPredictionStorage
            import redis

            prediction_ttl = int(os.getenv("ML_PREDICTION_TTL", "2592000"))

            # Create Redis client with production-optimized settings
            redis_client = redis.Redis.from_url(
                redis_url,
                decode_responses=True,
                socket_timeout=10,
                socket_connect_timeout=10,
                retry_on_timeout=True,
                health_check_interval=30,
                max_connections=20
            )

            storage = RedisPredictionStorage(
                redis_client=redis_client,
                prediction_ttl=prediction_ttl,
                verbose=verbose
            )

            # Determine environment name for logging
            env_name = "Production" if not redis_url.startswith("redis://localhost") else "Development"
            if verbose:
                # Mask sensitive parts of URL for logging
                safe_url = redis_url.split('@')[0] + '@***' if '@' in redis_url else redis_url.split('://')[0] + '://***'
                print(f"🗄️  Using Redis storage for {env_name} environment")
                print(f"   URL: {safe_url}")
                print(f"   TTL: {prediction_ttl}s ({prediction_ttl//86400} days)")

            return storage

        except ImportError as e:
            print(f"⚠️  Redis not available, falling back to in-memory storage: {e}")
            return InMemoryPredictionStorage()
        except Exception as e:
            print(f"⚠️  Redis connection failed, falling back to in-memory storage: {e}")
            return InMemoryPredictionStorage()

    else:
        # No Redis available - use in-memory storage
        if verbose:
            print(f"🧪 Using in-memory storage")
            print("   Create .env.redis file in project root with REDIS_URL to enable Redis")
        return InMemoryPredictionStorage()


def create_shadow_controller(
    storage: Optional[PredictionStorage] = None,
    policy: DecisionPolicy = DecisionPolicy.BALANCED,
    verbose: bool = False,
    enable_mlflow: bool = True
) -> ShadowController:
    """
    Create Shadow Controller with production-ready configuration.

    Args:
        storage: Custom storage implementation (if None, uses create_production_storage())
        policy: Initial decision policy
        verbose: Enable verbose logging
        enable_mlflow: Enable MLflow experiment tracking

    Returns:
        Configured Shadow Controller ready for production use
    """
    if storage is None:
        storage = create_production_storage()

    # Initialize ML components
    feature_engineer = BNPLFeatureEngineer()
    predictor = BNPLPredictor(mode="shadow")

    # Create controller with MLflow support
    controller = ShadowController(
        predictor=predictor,
        feature_engineer=feature_engineer,
        storage=storage,
        verbose=verbose,
        enable_mlflow=enable_mlflow
    )

    # Set decision policy
    controller.decision_manager.update_policy(policy)

    if verbose:
        print(f"🎮 Shadow Controller initialized")
        print(f"   Policy: {policy.value}")
        print(f"   Storage: {type(storage).__name__}")
        print(f"   Models: {list(predictor.models.keys())}")
        if controller.enable_mlflow:
            print(f"   MLflow: enabled")

    return controller