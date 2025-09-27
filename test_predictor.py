#!/usr/bin/env python3
"""
Test the BNPL multi-model predictor with different deployment modes
"""

import sys
sys.path.append('.')

from flit_ml.features.bnpl_feature_engineering import BNPLFeatureEngineer
from flit_ml.models.bnpl.predictor import BNPLPredictor

def test_predictor_modes():
    """Test all deployment modes of the BNPLPredictor."""

    # Sample transaction (using real API structure)
    sample_transaction = {
        "transaction_id": "txn_test_003",
        "customer_id": "cust_test_003",
        "amount": 500.0,
        "transaction_timestamp": "2025-09-26T16:00:00Z",
        "customer_credit_score_range": "good",
        "customer_age_bracket": "35-44",
        "customer_income_bracket": "75k-100k",
        "customer_verification_level": "verified",
        "customer_tenure_days": 300,
        "device_type": "tablet",
        "device_is_trusted": True,
        "product_category": "sports",
        "product_risk_category": "low",
        "risk_score": 0.45,
        "risk_level": "low",
        "risk_scenario": "low_risk_purchase",
        "payment_provider": "sezzle",
        "installment_count": 6,
        "payment_credit_limit": 2000,
        "price_comparison_time": 90.0,
        "purchase_context": "normal"
    }

    print("🧪 Testing BNPL Multi-Model Predictor...")
    print(f"Test transaction: ${sample_transaction['amount']} via {sample_transaction['payment_provider']}")

    # Generate features
    engineer = BNPLFeatureEngineer(client=None, verbose=False)
    features = engineer.engineer_single_transaction(sample_transaction)
    print(f"✅ Features generated: {features.shape}")

    # Test different modes
    test_modes = ["shadow", "champion", "ridge", "logistic", "elastic", "ensemble"]
    results = {}

    for mode in test_modes:
        print(f"\n--- Testing {mode.upper()} mode ---")

        try:
            # Initialize predictor
            predictor = BNPLPredictor(mode=mode, verbose=False)

            # Get model info
            info = predictor.get_model_info()
            print(f"Models loaded: {info['models_loaded']}")
            print(f"Champion: {info['champion']}")

            # Generate prediction
            prediction = predictor.predict(features)
            results[mode] = prediction

            print(f"Prediction result: {prediction}")

        except Exception as e:
            print(f"❌ Error in {mode} mode: {e}")
            import traceback
            traceback.print_exc()
            return False

    # Analyze results
    print(f"\n{'='*60}")
    print("📊 PREDICTION ANALYSIS")

    # Shadow mode should have all 4 predictions
    shadow_result = results.get("shadow", {})
    if "ridge" in shadow_result and "logistic" in shadow_result:
        print(f"Shadow mode predictions:")
        for model in ["ridge", "logistic", "elastic", "ensemble"]:
            if model in shadow_result:
                print(f"  {model}: {shadow_result[model]}")
        print(f"  Champion: {shadow_result.get('champion')}")

    # Compare champion vs specific model predictions
    champion_result = results.get("champion", {})
    ridge_result = results.get("ridge", {})

    if champion_result and ridge_result:
        champion_pred = champion_result.get("prediction")
        ridge_pred = ridge_result.get("prediction")
        print(f"\nChampion vs Ridge comparison:")
        print(f"  Champion mode: {champion_pred}")
        print(f"  Ridge mode: {ridge_pred}")
        print(f"  Match: {champion_pred == ridge_pred}")

    # Performance analysis
    print(f"\nInference times:")
    for mode, result in results.items():
        inference_time = result.get("inference_time_ms", "N/A")
        print(f"  {mode}: {inference_time}ms")

    return True

def test_end_to_end_pipeline():
    """Test complete end-to-end pipeline: JSON → Features → Prediction."""

    print(f"\n{'='*60}")
    print("🚀 END-TO-END PIPELINE TEST")

    # Real transaction JSON
    transaction = {
        "amount": 199.99,
        "transaction_timestamp": "2025-09-26T20:30:00Z",  # Evening
        "customer_credit_score_range": "fair",
        "customer_age_bracket": "18-24",
        "customer_income_bracket": "25k-50k",
        "customer_verification_level": "partial",
        "customer_tenure_days": 45,
        "device_type": "mobile",
        "device_is_trusted": False,
        "product_category": "clothing",
        "product_risk_category": "medium",
        "risk_score": 0.65,
        "risk_level": "medium",
        "risk_scenario": "impulse_purchase",
        "payment_provider": "afterpay",
        "installment_count": 4,
        "payment_credit_limit": 800,
        "price_comparison_time": 15.0,
        "purchase_context": "rushed"
    }

    print(f"Transaction: ${transaction['amount']} {transaction['product_category']}")
    print(f"Customer: {transaction['customer_age_bracket']}, {transaction['customer_credit_score_range']}")
    print(f"Context: {transaction['purchase_context']}, {transaction['risk_scenario']}")

    try:
        # Step 1: Feature Engineering
        engineer = BNPLFeatureEngineer(client=None, verbose=False)
        features = engineer.engineer_single_transaction(transaction)

        # Step 2: Multi-Model Prediction
        predictor = BNPLPredictor(mode="shadow", verbose=False)
        predictions = predictor.predict(features)

        # Step 3: Results
        print(f"\n🎯 PREDICTIONS:")
        for model in ["ridge", "logistic", "elastic", "ensemble"]:
            if model in predictions:
                pred = predictions[model]
                risk_level = "HIGH" if pred > 0.7 else "MEDIUM" if pred > 0.4 else "LOW"
                print(f"  {model.upper()}: {pred:.4f} ({risk_level} RISK)")

        print(f"\nChampion: {predictions.get('champion')}")
        print(f"Processing time: {predictions.get('inference_time_ms')}ms")

        return True

    except Exception as e:
        print(f"❌ End-to-end test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success1 = test_predictor_modes()
    success2 = test_end_to_end_pipeline()

    if success1 and success2:
        print(f"\n🎉 All tests passed! Multi-model predictor is ready for production.")
    else:
        print(f"\n❌ Some tests failed!")

    sys.exit(0 if (success1 and success2) else 1)