#!/usr/bin/env python3
"""
Test BNPL API endpoints for functionality and performance
"""

import sys
import asyncio
import json
import time
from datetime import datetime

sys.path.append('.')

from fastapi.testclient import TestClient
from flit_ml.api.main import app

# Test client
client = TestClient(app)

def test_api_endpoints():
    """Test all API endpoints for functionality and performance."""

    print("🧪 Testing BNPL API Endpoints...")

    # Sample transaction (using real API structure)
    sample_transaction = {
        "transaction_id": "txn_api_test_001",
        "customer_id": "cust_api_test_001",
        "amount": 450.0,
        "transaction_timestamp": "2025-09-27T15:30:00Z",
        "customer_credit_score_range": "fair",
        "customer_age_bracket": "25-34",
        "customer_income_bracket": "50k-75k",
        "customer_verification_level": "verified",
        "customer_tenure_days": 180,
        "device_type": "mobile",
        "device_is_trusted": True,
        "product_category": "electronics",
        "product_risk_category": "medium",
        "risk_score": 0.65,
        "risk_level": "medium",
        "risk_scenario": "impulse_purchase",
        "payment_provider": "klarna",
        "installment_count": 4,
        "payment_credit_limit": 1500.0,
        "price_comparison_time": 60.0,
        "purchase_context": "normal"
    }

    results = {}

    # Test 1: Root endpoint
    print("\n--- Test 1: Root Endpoint ---")
    try:
        response = client.get("/")
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Service: {data['service']}")
            print(f"Version: {data['version']}")
            results["root"] = "✅ PASS"
        else:
            results["root"] = "❌ FAIL"
    except Exception as e:
        print(f"❌ Error: {e}")
        results["root"] = "❌ FAIL"

    # Test 2: Global health check
    print("\n--- Test 2: Global Health Check ---")
    try:
        response = client.get("/health")
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Status: {data['status']}")
            results["global_health"] = "✅ PASS"
        else:
            results["global_health"] = "❌ FAIL"
    except Exception as e:
        print(f"❌ Error: {e}")
        results["global_health"] = "❌ FAIL"

    # Test 3: BNPL health check
    print("\n--- Test 3: BNPL Health Check ---")
    try:
        response = client.get("/v1/bnpl/health")
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Status: {data['status']}")
            print(f"Model Status: {data['model_status']}")
            print(f"Version: {data['version']}")
            results["bnpl_health"] = "✅ PASS"
        else:
            results["bnpl_health"] = "❌ FAIL"
    except Exception as e:
        print(f"❌ Error: {e}")
        results["bnpl_health"] = "❌ FAIL"

    # Test 4: Model info endpoint
    print("\n--- Test 4: Model Info ---")
    try:
        response = client.get("/v1/bnpl/models/info")
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Mode: {data['mode']}")
            print(f"Models loaded: {data['models_loaded']}")
            print(f"Champion: {data['champion']}")
            print(f"Feature count: {data['feature_count']}")
            results["model_info"] = "✅ PASS"
        else:
            results["model_info"] = "❌ FAIL"
    except Exception as e:
        print(f"❌ Error: {e}")
        results["model_info"] = "❌ FAIL"

    # Test 5: Risk assessment (main endpoint)
    print("\n--- Test 5: Risk Assessment ---")
    try:
        start_time = time.time()
        response = client.post("/v1/bnpl/risk-assessment", json=sample_transaction)
        request_time = (time.time() - start_time) * 1000

        print(f"Status: {response.status_code}")
        print(f"Request time: {request_time:.1f}ms")

        if response.status_code == 200:
            data = response.json()

            print(f"\n🎯 Risk Assessment Results:")
            print(f"Transaction ID: {data['transaction_id']}")
            print(f"Risk Level: {data['risk_level']}")
            print(f"Default Probability: {data['default_probability']:.4f}")
            print(f"Champion Model: {data['champion_model']}")

            print(f"\nModel Predictions:")
            for model, prediction in data['predictions'].items():
                if model not in ['champion', 'inference_time_ms']:
                    print(f"  {model}: {prediction:.4f}")

            print(f"\nPerformance:")
            print(f"  Total processing: {data['processing_time_ms']:.1f}ms")
            print(f"  Model inference: {data['model_inference_time_ms']:.1f}ms")
            print(f"  API overhead: {request_time - data['processing_time_ms']:.1f}ms")

            print(f"\nDeployment Info:")
            print(f"  Mode: {data['deployment_mode']}")
            print(f"  Version: {data['model_version']}")

            # Validate response structure
            required_fields = [
                'transaction_id', 'risk_level', 'default_probability',
                'champion_model', 'predictions', 'processing_time_ms'
            ]

            missing_fields = [field for field in required_fields if field not in data]
            if missing_fields:
                print(f"❌ Missing fields: {missing_fields}")
                results["risk_assessment"] = "❌ FAIL"
            else:
                print(f"✅ All required fields present")
                results["risk_assessment"] = "✅ PASS"

        else:
            print(f"❌ Failed with status {response.status_code}")
            print(f"Response: {response.text}")
            results["risk_assessment"] = "❌ FAIL"

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        results["risk_assessment"] = "❌ FAIL"

    # Test 6: Invalid input handling
    print("\n--- Test 6: Invalid Input Handling ---")
    try:
        invalid_transaction = sample_transaction.copy()
        invalid_transaction["amount"] = -100  # Invalid negative amount

        response = client.post("/v1/bnpl/risk-assessment", json=invalid_transaction)
        print(f"Status: {response.status_code}")

        if response.status_code == 422:  # Validation error
            print("✅ Correctly rejected invalid input")
            results["input_validation"] = "✅ PASS"
        else:
            print("❌ Failed to reject invalid input")
            results["input_validation"] = "❌ FAIL"

    except Exception as e:
        print(f"❌ Error: {e}")
        results["input_validation"] = "❌ FAIL"

    # Summary
    print(f"\n{'='*60}")
    print("📊 API TEST SUMMARY")
    print(f"{'='*60}")

    passed = sum(1 for result in results.values() if result == "✅ PASS")
    total = len(results)

    for test_name, result in results.items():
        print(f"{test_name:20s}: {result}")

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All API tests passed! Endpoints ready for production.")
        return True
    else:
        print("❌ Some API tests failed!")
        return False


if __name__ == "__main__":
    success = test_api_endpoints()
    sys.exit(0 if success else 1)