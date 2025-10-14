"""
Unit tests for Shadow Mode Controller

Tests the core functionality of experiment management, business decision logic,
and storage abstraction without external dependencies.
"""

import pytest
import asyncio
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock
import numpy as np

from flit_ml.core.shadow_controller import (
    ShadowController,
    ExperimentManager,
    DecisionManager,
    ExperimentConfig,
    ExperimentStatus,
    DecisionPolicy,
    PredictionLog,
    InMemoryPredictionStorage
)


class MockBNPLPredictor:
    """Mock predictor for testing."""

    def __init__(self, mode="shadow"):
        self.mode = mode
        self.model_version = "v0.1.0"
        self.models = {"ridge": "mock", "logistic": "mock", "elastic": "mock", "ensemble": "mock"}

    def predict(self, features):
        return {
            "ridge": 0.234,
            "logistic": 0.251,
            "elastic": 0.240,
            "ensemble": 0.242,
            "champion": "ridge",
            "inference_time_ms": 1.5
        }


class MockFeatureEngineer:
    """Mock feature engineer for testing."""

    def __init__(self):
        pass

    def engineer_single_transaction(self, transaction_data):
        # Return mock DataFrame-like object
        mock_features = Mock()
        mock_features.values = Mock()
        mock_features.values.flatten.return_value = np.array([1, 2, 3, 4, 5])
        return mock_features


class TestExperimentManager:
    """Test experiment management functionality."""

    @pytest.fixture
    def experiment_manager(self):
        return ExperimentManager(verbose=False)

    @pytest.fixture
    def sample_experiment_config(self):
        return ExperimentConfig(
            experiment_id="test_exp_001",
            name="Test Experiment",
            description="Testing experiment functionality",
            traffic_allocation={"champion": 0.7, "challenger": 0.3},
            models=["ridge", "logistic"],
            start_date=datetime.utcnow(),
            end_date=None,
            success_metric="approval_rate",
            min_sample_size=1000,
            created_by="test"
        )

    def test_create_experiment(self, experiment_manager, sample_experiment_config):
        """Test experiment creation."""
        experiment_id = experiment_manager.create_experiment(sample_experiment_config)

        assert experiment_id == "test_exp_001"
        assert experiment_id in experiment_manager.experiments
        assert experiment_manager.experiments[experiment_id].name == "Test Experiment"

    def test_experiment_config_validation(self):
        """Test experiment configuration validation."""
        # Test invalid traffic allocation
        with pytest.raises(ValueError, match="Traffic allocation must sum to 1.0"):
            ExperimentConfig(
                experiment_id="invalid_exp",
                name="Invalid",
                description="Invalid traffic allocation",
                traffic_allocation={"champion": 0.8, "challenger": 0.3},  # Sums to 1.1
                models=["ridge", "logistic"],
                start_date=datetime.utcnow(),
                end_date=None,
                success_metric="approval_rate",
                min_sample_size=1000
            )

    def test_start_stop_experiment(self, experiment_manager, sample_experiment_config):
        """Test experiment lifecycle management."""
        # Create experiment
        experiment_id = experiment_manager.create_experiment(sample_experiment_config)

        # Start experiment
        result = experiment_manager.start_experiment(experiment_id)
        assert result is True
        assert experiment_manager.active_experiment == experiment_id
        assert experiment_manager.experiments[experiment_id].status == ExperimentStatus.ACTIVE

        # Stop experiment
        result = experiment_manager.stop_experiment(experiment_id)
        assert result is True
        assert experiment_manager.active_experiment is None
        assert experiment_manager.experiments[experiment_id].status == ExperimentStatus.COMPLETED

    def test_traffic_assignment_deterministic(self, experiment_manager, sample_experiment_config):
        """Test that traffic assignment is deterministic for same customer."""
        experiment_manager.create_experiment(sample_experiment_config)

        customer_id = "cust_123"
        assignment1 = experiment_manager.assign_traffic(customer_id, sample_experiment_config)
        assignment2 = experiment_manager.assign_traffic(customer_id, sample_experiment_config)

        assert assignment1 == assignment2

    def test_traffic_assignment_distribution(self, experiment_manager, sample_experiment_config):
        """Test traffic assignment distribution over many customers."""
        experiment_manager.create_experiment(sample_experiment_config)

        assignments = []
        for i in range(1000):
            customer_id = f"cust_{i}"
            assignment = experiment_manager.assign_traffic(customer_id, sample_experiment_config)
            assignments.append(assignment)

        champion_count = assignments.count("champion")
        challenger_count = assignments.count("challenger")

        # Allow for statistical variance
        assert 0.65 <= champion_count / len(assignments) <= 0.75
        assert 0.25 <= challenger_count / len(assignments) <= 0.35

    def test_apply_experiment_logic_no_active_experiment(self, experiment_manager):
        """Test experiment logic when no experiment is active."""
        predictions = {"ridge": 0.234, "logistic": 0.251, "champion": "ridge"}
        transaction_data = {"customer_id": "cust_123"}

        result = experiment_manager.apply_experiment_logic(predictions, transaction_data)

        assert result["selected_model"] == "ridge"
        assert result["selected_prediction"] == 0.234
        assert result["experiment_id"] is None
        assert result["traffic_segment"] == "champion"

    def test_apply_experiment_logic_with_active_experiment(self, experiment_manager, sample_experiment_config):
        """Test experiment logic with active experiment."""
        experiment_manager.create_experiment(sample_experiment_config)
        experiment_manager.start_experiment(sample_experiment_config.experiment_id)

        predictions = {"ridge": 0.234, "logistic": 0.251, "champion": "ridge"}
        transaction_data = {"customer_id": "cust_123"}

        result = experiment_manager.apply_experiment_logic(predictions, transaction_data)

        assert result["selected_model"] in ["ridge", "logistic"]
        assert result["experiment_id"] == "test_exp_001"
        assert result["traffic_segment"] in ["champion", "challenger"]


class TestDecisionManager:
    """Test business decision logic."""

    @pytest.fixture
    def decision_manager(self):
        return DecisionManager(initial_policy=DecisionPolicy.BALANCED)

    def test_business_decision_balanced_policy(self, decision_manager):
        """Test business decisions with balanced policy."""
        # Test low risk (approve)
        decision, risk_level = decision_manager.make_business_decision(0.3, "ridge")
        assert decision == "approve"
        assert risk_level == "LOW"

        # Test medium risk (manual review)
        decision, risk_level = decision_manager.make_business_decision(0.5, "ridge")
        assert decision == "manual_review"
        assert risk_level == "MEDIUM"

        # Test high risk (deny)
        decision, risk_level = decision_manager.make_business_decision(0.8, "ridge")
        assert decision == "deny"
        assert risk_level == "HIGH"

    def test_policy_update(self, decision_manager):
        """Test policy updates."""
        # Test conservative policy (high=0.5, medium=0.25)
        decision_manager.update_policy(DecisionPolicy.CONSERVATIVE)
        decision, risk_level = decision_manager.make_business_decision(0.3, "ridge")
        assert decision == "manual_review"  # Above medium threshold (0.25)

        # Test aggressive policy (high=0.8, medium=0.5)
        decision_manager.update_policy(DecisionPolicy.AGGRESSIVE)
        decision, risk_level = decision_manager.make_business_decision(0.4, "ridge")
        assert decision == "approve"  # Below medium threshold (0.5)

    def test_custom_thresholds(self, decision_manager):
        """Test custom threshold configuration."""
        custom_thresholds = {"high": 0.9, "medium": 0.6}
        decision_manager.update_policy(DecisionPolicy.CUSTOM, custom_thresholds)

        # Test with custom thresholds
        decision, risk_level = decision_manager.make_business_decision(0.7, "ridge")
        assert decision == "manual_review"  # Above custom medium threshold

        thresholds = decision_manager.get_current_thresholds()
        assert thresholds["high"] == 0.9
        assert thresholds["medium"] == 0.6


class TestInMemoryPredictionStorage:
    """Test in-memory storage implementation."""

    @pytest.fixture
    def storage(self):
        return InMemoryPredictionStorage(max_size=100)

    @pytest.fixture
    def sample_prediction_log(self):
        return PredictionLog(
            prediction_id="pred_001",
            transaction_id="tx_001",
            customer_id="cust_001",
            timestamp=datetime.utcnow(),
            model_predictions={"ridge": 0.234, "logistic": 0.251},
            selected_model="ridge",
            selected_prediction=0.234,
            business_decision="approve",
            risk_level="LOW",
            decision_policy="balanced",
            risk_thresholds={"high": 0.7, "medium": 0.4}
        )

    def test_store_and_retrieve_prediction(self, storage, sample_prediction_log):
        """Test storing and retrieving predictions."""
        storage.store_prediction(sample_prediction_log)

        recent_predictions = storage.get_recent_predictions(hours=1)
        assert len(recent_predictions) == 1
        assert recent_predictions[0].prediction_id == "pred_001"

    def test_get_recent_predictions_time_filter(self, storage):
        """Test time-based filtering of predictions."""
        # Create predictions with different timestamps
        old_prediction = PredictionLog(
            prediction_id="pred_old",
            transaction_id="tx_old",
            customer_id="cust_old",
            timestamp=datetime.utcnow() - timedelta(hours=2),
            model_predictions={"ridge": 0.234},
            selected_model="ridge",
            selected_prediction=0.234,
            business_decision="approve",
            risk_level="LOW",
            decision_policy="balanced",
            risk_thresholds={"high": 0.7, "medium": 0.4}
        )

        recent_prediction = PredictionLog(
            prediction_id="pred_recent",
            transaction_id="tx_recent",
            customer_id="cust_recent",
            timestamp=datetime.utcnow(),
            model_predictions={"ridge": 0.234},
            selected_model="ridge",
            selected_prediction=0.234,
            business_decision="approve",
            risk_level="LOW",
            decision_policy="balanced",
            risk_thresholds={"high": 0.7, "medium": 0.4}
        )

        storage.store_prediction(old_prediction)
        storage.store_prediction(recent_prediction)

        # Get predictions from last 1 hour
        recent_predictions = storage.get_recent_predictions(hours=1)
        assert len(recent_predictions) == 1
        assert recent_predictions[0].prediction_id == "pred_recent"

    def test_get_experiment_data(self, storage):
        """Test retrieving experiment-specific data."""
        prediction1 = PredictionLog(
            prediction_id="pred_001",
            transaction_id="tx_001",
            customer_id="cust_001",
            timestamp=datetime.utcnow(),
            model_predictions={"ridge": 0.234},
            selected_model="ridge",
            selected_prediction=0.234,
            business_decision="approve",
            risk_level="LOW",
            decision_policy="balanced",
            risk_thresholds={"high": 0.7, "medium": 0.4},
            experiment_id="exp_001"
        )

        prediction2 = PredictionLog(
            prediction_id="pred_002",
            transaction_id="tx_002",
            customer_id="cust_002",
            timestamp=datetime.utcnow(),
            model_predictions={"ridge": 0.234},
            selected_model="ridge",
            selected_prediction=0.234,
            business_decision="approve",
            risk_level="LOW",
            decision_policy="balanced",
            risk_thresholds={"high": 0.7, "medium": 0.4},
            experiment_id="exp_002"
        )

        storage.store_prediction(prediction1)
        storage.store_prediction(prediction2)

        exp_data = storage.get_experiment_data("exp_001")
        assert len(exp_data) == 1
        assert exp_data[0].prediction_id == "pred_001"

    def test_max_size_limit(self, storage):
        """Test that storage respects maximum size limit."""
        # Add more predictions than max_size
        for i in range(150):
            prediction = PredictionLog(
                prediction_id=f"pred_{i:03d}",
                transaction_id=f"tx_{i:03d}",
                customer_id=f"cust_{i:03d}",
                timestamp=datetime.utcnow(),
                model_predictions={"ridge": 0.234},
                selected_model="ridge",
                selected_prediction=0.234,
                business_decision="approve",
                risk_level="LOW",
                decision_policy="balanced",
                risk_thresholds={"high": 0.7, "medium": 0.4}
            )
            storage.store_prediction(prediction)

        # Should only keep max_size predictions
        recent_predictions = storage.get_recent_predictions(hours=24)
        assert len(recent_predictions) == 100


class TestShadowController:
    """Test complete Shadow Controller functionality."""

    @pytest.fixture
    def shadow_controller(self):
        return ShadowController(
            predictor=MockBNPLPredictor(),
            feature_engineer=MockFeatureEngineer(),
            storage=InMemoryPredictionStorage(),
            verbose=False
        )

    @pytest.fixture
    def sample_transaction_data(self):
        return {
            "transaction_id": "tx_001",
            "customer_id": "cust_001",
            "amount": 299.99,
            "transaction_timestamp": "2025-09-27T14:30:00Z",
            "customer_credit_score_range": "good",
            "customer_age_bracket": "25-34",
            "customer_income_bracket": "50k-75k",
            "customer_verification_level": "verified",
            "customer_tenure_days": 365,
            "device_type": "mobile",
            "device_is_trusted": True,
            "product_category": "electronics",
            "product_risk_category": "medium",
            "risk_score": 0.234,
            "risk_level": "low",
            "risk_scenario": "low_risk_purchase",
            "payment_provider": "klarna",
            "installment_count": 4,
            "payment_credit_limit": 1500.0,
            "price_comparison_time": 45.2,
            "purchase_context": "normal"
        }

    def test_assess_risk_basic_functionality(self, shadow_controller, sample_transaction_data):
        """Test basic risk assessment functionality."""
        result = shadow_controller.assess_risk_with_logging(sample_transaction_data, log_prediction=False)

        # Check response structure
        assert "prediction_id" in result
        assert "business_decision" in result
        assert "risk_level" in result
        assert "selected_model" in result
        assert "all_predictions" in result
        assert "processing_time_ms" in result

        # Check business decision
        assert result["business_decision"] in ["approve", "deny", "manual_review"]
        assert result["risk_level"] in ["LOW", "MEDIUM", "HIGH"]

        # Check model information
        assert result["selected_model"] in ["ridge", "logistic", "elastic", "ensemble"]
        assert "ridge" in result["all_predictions"]

    def test_assess_risk_with_logging(self, shadow_controller, sample_transaction_data):
        """Test risk assessment with prediction logging."""
        result = shadow_controller.assess_risk_with_logging(sample_transaction_data, log_prediction=True)

        # Give async operations time to complete
        time.sleep(0.1)

        # Check that prediction was logged
        recent_predictions = shadow_controller.storage.get_recent_predictions(hours=1)
        assert len(recent_predictions) == 1

        logged_prediction = recent_predictions[0]
        assert logged_prediction.transaction_id == "tx_001"
        assert logged_prediction.customer_id == "cust_001"
        assert logged_prediction.selected_model in ["ridge", "logistic", "elastic", "ensemble"]

    def test_assess_risk_with_experiment(self, shadow_controller, sample_transaction_data):
        """Test risk assessment with active experiment."""
        # Create and start experiment
        experiment_config = ExperimentConfig(
            experiment_id="test_exp",
            name="Test Experiment",
            description="Testing",
            traffic_allocation={"champion": 0.5, "challenger": 0.5},
            models=["ridge", "logistic"],
            start_date=datetime.utcnow(),
            end_date=None,
            success_metric="approval_rate",
            min_sample_size=100
        )

        shadow_controller.experiment_manager.create_experiment(experiment_config)
        shadow_controller.experiment_manager.start_experiment("test_exp")

        result = shadow_controller.assess_risk_with_logging(sample_transaction_data, log_prediction=False)

        # Check experiment information
        assert result["experiment_id"] == "test_exp"
        assert result["traffic_segment"] in ["champion", "challenger"]
        assert result["selected_model"] in ["ridge", "logistic"]

    def test_get_recent_performance(self, shadow_controller, sample_transaction_data):
        """Test performance metrics calculation."""
        # Generate some test predictions
        for i in range(10):
            transaction_data = sample_transaction_data.copy()
            transaction_data["customer_id"] = f"cust_{i:03d}"
            transaction_data["transaction_id"] = f"tx_{i:03d}"
            shadow_controller.assess_risk_with_logging(transaction_data, log_prediction=True)

        time.sleep(0.1)  # Allow async operations to complete

        performance = shadow_controller.get_recent_performance(hours=1)

        assert performance["total_predictions"] == 10
        assert "avg_processing_time_ms" in performance
        assert "approval_rate" in performance
        assert "manual_review_rate" in performance
        assert "denial_rate" in performance
        assert "models_used" in performance

    def test_get_experiment_analysis(self, shadow_controller, sample_transaction_data):
        """Test experiment analysis functionality."""
        # Create experiment
        experiment_config = ExperimentConfig(
            experiment_id="analysis_exp",
            name="Analysis Test",
            description="Testing analysis",
            traffic_allocation={"champion": 0.5, "challenger": 0.5},
            models=["ridge", "logistic"],
            start_date=datetime.utcnow(),
            end_date=None,
            success_metric="approval_rate",
            min_sample_size=10
        )

        shadow_controller.experiment_manager.create_experiment(experiment_config)
        shadow_controller.experiment_manager.start_experiment("analysis_exp")

        # Generate predictions for the experiment
        for i in range(20):
            transaction_data = sample_transaction_data.copy()
            transaction_data["customer_id"] = f"analysis_cust_{i:03d}"
            transaction_data["transaction_id"] = f"analysis_tx_{i:03d}"
            shadow_controller.assess_risk_with_logging(transaction_data, log_prediction=True)

        time.sleep(0.1)  # Allow async operations to complete

        analysis = shadow_controller.get_experiment_analysis("analysis_exp", min_sample_size=10)

        assert analysis["status"] == "sufficient_data"
        assert analysis["total_sample_size"] >= 10
        assert "segment_metrics" in analysis

    def test_get_controller_info(self, shadow_controller):
        """Test controller status information."""
        info = shadow_controller.get_controller_info()

        assert info["status"] == "operational"
        assert info["predictor_mode"] == "shadow"
        assert "models_loaded" in info
        assert "current_policy" in info
        assert "decision_thresholds" in info
        assert "storage_type" in info
        assert info["storage_type"] == "InMemoryPredictionStorage"

    def test_error_handling(self, shadow_controller):
        """Test error handling with invalid input."""
        # Test with missing required fields that will cause feature engineering to fail
        invalid_data = {"transaction_id": "tx_001"}

        # Create a mock feature engineer that raises an exception for invalid data
        def mock_engineer_that_fails(transaction_data):
            if len(transaction_data) < 5:  # Not enough fields
                raise ValueError("Missing required fields for feature engineering")
            return shadow_controller.feature_engineer.engineer_single_transaction(transaction_data)

        shadow_controller.feature_engineer.engineer_single_transaction = mock_engineer_that_fails

        with pytest.raises(Exception):
            shadow_controller.assess_risk_with_logging(invalid_data)


class TestAsyncOperations:
    """Test asynchronous operations."""

    @pytest.fixture
    def shadow_controller(self):
        return ShadowController(
            predictor=MockBNPLPredictor(),
            feature_engineer=MockFeatureEngineer(),
            storage=InMemoryPredictionStorage(),
            verbose=False
        )

    @pytest.mark.asyncio
    async def test_async_logging(self, shadow_controller):
        """Test that async logging doesn't block main operations."""
        prediction_log = PredictionLog(
            prediction_id="async_test",
            transaction_id="tx_async",
            customer_id="cust_async",
            timestamp=datetime.utcnow(),
            model_predictions={"ridge": 0.234},
            selected_model="ridge",
            selected_prediction=0.234,
            business_decision="approve",
            risk_level="LOW",
            decision_policy="balanced",
            risk_thresholds={"high": 0.7, "medium": 0.4}
        )

        # Test async logging
        await shadow_controller._log_prediction_async(prediction_log)

        # Verify prediction was stored
        recent_predictions = shadow_controller.storage.get_recent_predictions(hours=1)
        assert len(recent_predictions) == 1
        assert recent_predictions[0].prediction_id == "async_test"


if __name__ == "__main__":
    pytest.main([__file__])