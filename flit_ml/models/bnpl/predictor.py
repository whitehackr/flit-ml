"""
BNPL Multi-Model Predictor

Flexible predictor supporting multiple deployment modes:
- Shadow mode: All 4 models for comparison
- Champion mode: Best performing model only
- Specific model: Individual model selection
- A/B testing: Custom model combinations
"""

import joblib
import pandas as pd
import numpy as np
import logging
import time
from typing import Dict, Union, List, Optional
from pathlib import Path


class BNPLPredictor:
    """
    Flexible multi-model predictor for BNPL default risk assessment.

    Supports multiple deployment modes for different operational needs:
    - Shadow deployment for model comparison
    - Champion-only for optimized production
    - A/B testing for controlled experiments
    """

    def __init__(self, mode: str = "shadow", model_version: str = "v0.1.0", verbose: bool = True):
        """
        Initialize BNPL predictor with specified deployment mode.

        Args:
            mode: Deployment mode
                - "shadow": Load all 4 models for comparison
                - "champion": Load only champion model (ridge)
                - "ridge", "logistic", "elastic", "ensemble": Load specific model
            model_version: Model version to load (default: v0.1.0)
            verbose: Enable verbose logging
        """
        self.mode = mode
        self.model_version = model_version
        self.verbose = verbose

        # Setup logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO if verbose else logging.WARNING)

        # Model paths
        self.model_dir = Path("models/production")

        # Initialize containers
        self.models = {}
        self.preprocessor = None
        self.metadata = None

        # Load artifacts
        self._load_artifacts()

    def _load_artifacts(self):
        """Load model artifacts based on deployment mode."""
        start_time = time.time()

        if self.verbose:
            print(f"🔄 Loading BNPL predictor in {self.mode} mode...")

        # Load metadata
        metadata_path = self.model_dir / f"bnpl_multi_model_metadata_{self.model_version}.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")

        import json
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)

        # Load preprocessor
        preprocessor_path = self.model_dir / f"bnpl_preprocessor_{self.model_version}.joblib"
        if not preprocessor_path.exists():
            raise FileNotFoundError(f"Preprocessor not found: {preprocessor_path}")

        self.preprocessor = joblib.load(preprocessor_path)

        # Load models based on mode
        if self.mode == "shadow":
            self._load_all_models()
        elif self.mode == "champion":
            champion_model = self.metadata["deployment_strategy"]["champion"]
            self._load_specific_model(champion_model)
        elif self.mode in ["ridge", "logistic", "elastic", "ensemble"]:
            self._load_specific_model(self.mode)
        else:
            raise ValueError(f"Unknown mode: {self.mode}. Use 'shadow', 'champion', or specific model name.")

        load_time = (time.time() - start_time) * 1000

        if self.verbose:
            models_loaded = list(self.models.keys())
            print(f"✅ Loaded {len(models_loaded)} model(s) in {load_time:.1f}ms: {models_loaded}")

    def _load_all_models(self):
        """Load all 4 models for shadow mode."""
        model_names = ["ridge", "logistic", "elastic", "ensemble"]

        for model_name in model_names:
            self._load_specific_model(model_name)

    def _load_specific_model(self, model_name: str):
        """Load a specific model by name."""
        model_path = self.model_dir / f"bnpl_{model_name}_{self.model_version}.joblib"

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        self.models[model_name] = joblib.load(model_path)

    def predict(self, features: pd.DataFrame) -> Dict:
        """
        Generate predictions using loaded models.

        Args:
            features: DataFrame with 36 features in correct order (from feature engineering)

        Returns:
            Dictionary with predictions based on deployment mode:
            - Shadow mode: {"ridge": 0.23, "logistic": 0.25, "elastic": 0.24, "ensemble": 0.22, "champion": "ridge"}
            - Champion mode: {"prediction": 0.23, "model": "ridge"}
            - Specific model: {"prediction": 0.23, "model": "ridge"}
        """
        start_time = time.time()

        # Validate input
        if features.shape[1] != 36:
            raise ValueError(f"Expected 36 features, got {features.shape[1]}")

        # Apply preprocessing (scaling)
        features_processed = self.preprocessor.transform(features)

        # Generate predictions
        predictions = {}

        for model_name, model in self.models.items():
            # Handle different model types
            if model_name == "ensemble":
                # Special handling for VotingClassifier with mixed estimator types
                predictions[model_name] = self._predict_ensemble(model, features_processed)
            elif hasattr(model, 'predict_proba'):
                # Models with probability output (LogisticRegression, ElasticNet)
                pred_proba = model.predict_proba(features_processed)[0]
                default_probability = pred_proba[1] if len(pred_proba) > 1 else pred_proba[0]
                predictions[model_name] = round(float(default_probability), 4)
            elif hasattr(model, 'decision_function'):
                # Models with decision function (RidgeClassifier)
                decision_score = model.decision_function(features_processed)[0]
                # Convert decision function to probability-like score using sigmoid
                default_probability = 1 / (1 + np.exp(-decision_score))
                predictions[model_name] = round(float(default_probability), 4)
            else:
                # Fallback: binary prediction converted to probability
                binary_pred = model.predict(features_processed)[0]
                default_probability = float(binary_pred)
                predictions[model_name] = round(float(default_probability), 4)

        # Format response based on mode
        if self.mode == "shadow":
            result = predictions.copy()
            # Add champion indicator
            champion_model = self.metadata["deployment_strategy"]["champion"]
            result["champion"] = champion_model
        else:
            # Single model mode
            model_name = list(self.models.keys())[0]
            result = {
                "prediction": predictions[model_name],
                "model": model_name
            }

        # Add timing
        prediction_time = (time.time() - start_time) * 1000
        result["inference_time_ms"] = round(prediction_time, 2)

        if self.verbose:
            print(f"🎯 Prediction completed in {prediction_time:.1f}ms")

        return result

    def _predict_ensemble(self, ensemble_model, features_processed):
        """
        Custom ensemble prediction for mixed estimator types.

        Averages only models with calibrated probability outputs.
        See docs/models/bnpl0925_known_issues.md #4 for calibration limitation.

        Includes: LogisticRegression models (logistic + elastic)
        Excludes: RidgeClassifier (uncalibrated decision_function)
        """
        calibrated_predictions = []

        for estimator in ensemble_model.estimators_:
            if hasattr(estimator, 'predict_proba'):
                # LogisticRegression models with calibrated probabilities
                pred_proba = estimator.predict_proba(features_processed)[0]
                default_prob = pred_proba[1] if len(pred_proba) > 1 else pred_proba[0]
                calibrated_predictions.append(default_prob)
            # Skip RidgeClassifier (only has uncalibrated decision_function)

        if not calibrated_predictions:
            raise ValueError("No models with calibrated probabilities found in ensemble")

        # Average only calibrated LogisticRegression predictions
        ensemble_prediction = np.mean(calibrated_predictions)
        return round(float(ensemble_prediction), 4)

    def get_model_info(self) -> Dict:
        """Get information about loaded models."""
        return {
            "mode": self.mode,
            "version": self.model_version,
            "models_loaded": list(self.models.keys()),
            "champion": self.metadata["deployment_strategy"]["champion"],
            "feature_count": self.metadata["shared_artifacts"]["total_features"]
        }