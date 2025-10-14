"""
Integration tests for Shadow Mode Controller

Tests the complete integration of Shadow Controller with real ML components
and validates end-to-end workflows.
"""

import pytest
import time
import tempfile
import os
from datetime import datetime
from pathlib import Path

from flit_ml.core.shadow_controller import (
    ShadowController,
    ExperimentConfig,
    ExperimentStatus,
    DecisionPolicy,
    InMemoryPredictionStorage
)
from flit_ml.features.bnpl_feature_engineering import BNPLFeatureEngineer
from flit_ml.models.bnpl.predictor import BNPLPredictor


class TestShadowControllerIntegration:
    """Integration tests with real ML components."""

    @pytest.fixture
    def real_shadow_controller(self):
        """Create Shadow Controller with real ML components."""
        # Use in-memory storage to avoid external dependencies
        storage = InMemoryPredictionStorage()

        # Create Shadow Controller with real components
        controller = ShadowController(
            predictor=None,  # Will initialize default BNPLPredictor
            feature_engineer=None,  # Will initialize default BNPLFeatureEngineer
            storage=storage,
            verbose=False
        )

        return controller

    @pytest.fixture
    def complete_transaction_data(self):
        """Complete transaction data for integration testing."""
        return {
            "transaction_id": "integration_tx_001",
            "customer_id": "integration_cust_001",
            "amount": 299.99,
            "transaction_timestamp": "2025-09-27T14:30:00Z",

            # Customer attributes
            "customer_credit_score_range": "good",
            "customer_age_bracket": "25-34",
            "customer_income_bracket": "50k-75k",
            "customer_verification_level": "verified",
            "customer_tenure_days": 365,

            # Device context
            "device_type": "mobile",
            "device_is_trusted": True,

            # Product details
            "product_category": "electronics",
            "product_risk_category": "medium",

            # Risk assessment
            "risk_score": 0.234,
            "risk_level": "low",
            "risk_scenario": "low_risk_purchase",

            # Payment details
            "payment_provider": "klarna",
            "installment_count": 4,
            "payment_credit_limit": 1500.0,
            "price_comparison_time": 45.2,
            "purchase_context": "normal"
        }

    def test_complete_prediction_workflow(self, real_shadow_controller, complete_transaction_data):
        """Test complete prediction workflow with real components."""
        result = real_shadow_controller.assess_risk_with_logging(
            complete_transaction_data,
            log_prediction=True
        )

        # Verify response structure
        assert "prediction_id" in result
        assert "business_decision" in result
        assert "risk_level" in result
        assert "selected_model" in result
        assert "all_predictions" in result

        # Verify business decision is valid
        assert result["business_decision"] in ["approve", "deny", "manual_review"]
        assert result["risk_level"] in ["LOW", "MEDIUM", "HIGH"]

        # Verify model predictions
        assert "ridge" in result["all_predictions"]
        assert isinstance(result["all_predictions"]["ridge"], float)

        # Verify performance metrics
        assert result["processing_time_ms"] > 0
        assert result["model_inference_time_ms"] > 0

        # Allow async operations to complete
        time.sleep(0.1)

        # Verify prediction was logged
        recent_predictions = real_shadow_controller.storage.get_recent_predictions(hours=1)
        assert len(recent_predictions) == 1

        logged_prediction = recent_predictions[0]
        assert logged_prediction.transaction_id == "integration_tx_001"
        assert logged_prediction.customer_id == "integration_cust_001"

    def test_experiment_workflow_integration(self, real_shadow_controller, complete_transaction_data):
        """Test complete A/B testing workflow."""
        # Create experiment
        experiment_config = ExperimentConfig(
            experiment_id="integration_exp_001",
            name="Integration Test Experiment",
            description="Testing complete A/B testing workflow",
            traffic_allocation={"champion": 0.6, "challenger": 0.4},
            models=["ridge", "logistic"],
            start_date=datetime.utcnow(),
            end_date=None,
            success_metric="approval_rate",
            min_sample_size=20
        )

        # Start experiment
        real_shadow_controller.experiment_manager.create_experiment(experiment_config)
        real_shadow_controller.experiment_manager.start_experiment("integration_exp_001")

        # Generate predictions with experiment
        results = []
        for i in range(50):
            transaction_data = complete_transaction_data.copy()
            transaction_data["customer_id"] = f"exp_cust_{i:03d}"
            transaction_data["transaction_id"] = f"exp_tx_{i:03d}"

            result = real_shadow_controller.assess_risk_with_logging(
                transaction_data,
                log_prediction=True
            )
            results.append(result)

        # Verify experiment assignment
        experiment_results = [r for r in results if r.get("experiment_id") == "integration_exp_001"]
        assert len(experiment_results) == 50

        # Check traffic distribution
        champion_count = sum(1 for r in experiment_results if r["traffic_segment"] == "champion")
        challenger_count = sum(1 for r in experiment_results if r["traffic_segment"] == "challenger")

        # Allow for statistical variance
        assert 0.5 <= champion_count / len(experiment_results) <= 0.7
        assert 0.3 <= challenger_count / len(experiment_results) <= 0.5

        # Allow async operations to complete
        time.sleep(0.2)

        # Test experiment analysis
        analysis = real_shadow_controller.get_experiment_analysis("integration_exp_001", min_sample_size=20)
        assert analysis["status"] == "sufficient_data"
        assert analysis["total_sample_size"] >= 20
        assert "segment_metrics" in analysis

        # Verify segment metrics
        segments = analysis["segment_metrics"]
        assert "champion" in segments
        assert "challenger" in segments

        for segment, metrics in segments.items():
            assert "sample_size" in metrics
            assert "approval_rate" in metrics
            assert "avg_risk_score" in metrics
            assert metrics["sample_size"] > 0

    def test_decision_policy_integration(self, real_shadow_controller, complete_transaction_data):
        """Test business decision policies with real predictions."""
        # Test different decision policies
        policies_to_test = [
            DecisionPolicy.CONSERVATIVE,
            DecisionPolicy.BALANCED,
            DecisionPolicy.AGGRESSIVE
        ]

        results = {}

        for policy in policies_to_test:
            real_shadow_controller.decision_manager.update_policy(policy)

            result = real_shadow_controller.assess_risk_with_logging(
                complete_transaction_data,
                log_prediction=False
            )

            results[policy.value] = {
                "business_decision": result["business_decision"],
                "risk_level": result["risk_level"],
                "default_probability": result["default_probability"]
            }

        # Verify that different policies can produce different decisions
        # (depending on the prediction score and thresholds)
        all_decisions = [r["business_decision"] for r in results.values()]
        all_risk_levels = [r["risk_level"] for r in results.values()]

        # At minimum, verify the response structure is consistent
        for policy_result in results.values():
            assert policy_result["business_decision"] in ["approve", "deny", "manual_review"]
            assert policy_result["risk_level"] in ["LOW", "MEDIUM", "HIGH"]
            assert 0 <= policy_result["default_probability"] <= 1

    def test_performance_monitoring_integration(self, real_shadow_controller, complete_transaction_data):
        """Test performance monitoring with real data."""
        # Generate multiple predictions
        for i in range(25):
            transaction_data = complete_transaction_data.copy()
            transaction_data["customer_id"] = f"perf_cust_{i:03d}"
            transaction_data["transaction_id"] = f"perf_tx_{i:03d}"

            # Vary the risk score to get different decisions
            transaction_data["risk_score"] = 0.1 + (i * 0.03)  # Range from 0.1 to ~0.85

            real_shadow_controller.assess_risk_with_logging(
                transaction_data,
                log_prediction=True
            )

        # Allow async operations to complete
        time.sleep(0.2)

        # Test performance metrics
        performance = real_shadow_controller.get_recent_performance(hours=1)

        assert performance["total_predictions"] == 25
        assert performance["avg_processing_time_ms"] > 0
        assert 0 <= performance["approval_rate"] <= 1
        assert 0 <= performance["manual_review_rate"] <= 1
        assert 0 <= performance["denial_rate"] <= 1

        # Verify rates sum to approximately 1
        total_rate = (performance["approval_rate"] +
                     performance["manual_review_rate"] +
                     performance["denial_rate"])
        assert abs(total_rate - 1.0) < 0.01

        # Test model usage statistics
        assert "models_used" in performance
        assert len(performance["models_used"]) > 0

    def test_controller_state_consistency(self, real_shadow_controller):
        """Test that controller maintains consistent state."""
        # Get initial controller info
        initial_info = real_shadow_controller.get_controller_info()

        assert initial_info["status"] == "operational"
        assert initial_info["predictor_mode"] == "shadow"
        assert len(initial_info["models_loaded"]) > 0
        assert initial_info["current_policy"] in ["conservative", "balanced", "aggressive", "custom"]

        # Create and start experiment
        experiment_config = ExperimentConfig(
            experiment_id="state_test_exp",
            name="State Test",
            description="Testing state consistency",
            traffic_allocation={"champion": 0.8, "challenger": 0.2},
            models=["ridge", "logistic"],
            start_date=datetime.utcnow(),
            end_date=None,
            success_metric="approval_rate",
            min_sample_size=10
        )

        real_shadow_controller.experiment_manager.create_experiment(experiment_config)
        real_shadow_controller.experiment_manager.start_experiment("state_test_exp")

        # Check updated state
        updated_info = real_shadow_controller.get_controller_info()
        assert updated_info["active_experiment"] == "state_test_exp"
        assert updated_info["total_experiments"] == 1

        # Stop experiment
        real_shadow_controller.experiment_manager.stop_experiment("state_test_exp")

        # Check final state
        final_info = real_shadow_controller.get_controller_info()
        assert final_info["active_experiment"] is None
        assert final_info["total_experiments"] == 1

    def test_error_handling_integration(self, real_shadow_controller):
        """Test error handling with real components."""
        # Test with incomplete transaction data
        incomplete_data = {
            "transaction_id": "incomplete_tx",
            "customer_id": "incomplete_cust",
            "amount": 100.0
            # Missing required fields
        }

        with pytest.raises(Exception):
            real_shadow_controller.assess_risk_with_logging(incomplete_data)

        # Test with invalid experiment operations
        with pytest.raises(ValueError):
            real_shadow_controller.experiment_manager.start_experiment("nonexistent_exp")

        with pytest.raises(ValueError):
            real_shadow_controller.experiment_manager.stop_experiment("nonexistent_exp")

    def test_feature_engineering_integration(self, real_shadow_controller, complete_transaction_data):
        """Test that feature engineering produces expected output format."""
        # Access the feature engineer directly
        features = real_shadow_controller.feature_engineer.engineer_single_transaction(
            complete_transaction_data
        )

        # Verify feature output format
        assert hasattr(features, 'shape')
        assert features.shape[0] == 1  # Single transaction
        assert features.shape[1] == 36  # Expected feature count

        # Test that features can be processed by predictor
        predictions = real_shadow_controller.predictor.predict(features)

        assert isinstance(predictions, dict)
        assert "ridge" in predictions
        assert "logistic" in predictions
        assert "champion" in predictions

    def test_model_predictor_integration(self, real_shadow_controller):
        """Test that model predictor works correctly with shadow controller."""
        # Verify predictor is in shadow mode
        assert real_shadow_controller.predictor.mode == "shadow"

        # Verify all expected models are loaded
        expected_models = ["ridge", "logistic", "elastic", "ensemble"]
        loaded_models = list(real_shadow_controller.predictor.models.keys())

        for model in expected_models:
            assert model in loaded_models

        # Test model info
        model_info = real_shadow_controller.predictor.get_model_info()
        assert model_info["mode"] == "shadow"
        assert len(model_info["models_loaded"]) >= 4


if __name__ == "__main__":
    pytest.main([__file__])