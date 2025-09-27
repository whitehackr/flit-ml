#!/usr/bin/env python3
"""
Quick test for single transaction feature engineering
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from flit_ml.features.bnpl_feature_engineering import BNPLFeatureEngineer
import pandas as pd
from datetime import datetime

def test_single_transaction():
    """Test the single transaction feature engineering with real API JSON structure."""

    # Real transaction JSON structure from BigQuery json_body field
    real_api_transaction = {
        "transaction_id": "txn_test_001",
        "customer_id": "cust_test_001",
        "product_id": "prod_test_001",
        "device_id": "cust_test_001_device_1",
        "payment_method_id": "cust_test_001_pm_1",
        "amount": 299.99,
        "currency": "USD",
        "status": "completed",
        "purchase_context": "normal",
        "time_on_site_seconds": 150,
        "risk_scenario": "impulse_purchase",
        "risk_score": 0.75,
        "risk_level": "medium",
        "installment_count": 4,
        "first_payment_amount": 75.0,
        "payment_frequency": "bi_weekly",
        "checkout_speed": "normal",
        "cart_abandonment_count": 1,
        "price_comparison_time": 45,
        "economic_stress_factor": 0.2,
        "will_default": False,
        "days_to_first_missed_payment": None,
        "customer_age_bracket": "25-34",
        "customer_tenure_days": 120,
        "customer_income_bracket": "50k-75k",
        "customer_credit_score_range": "good",
        "customer_verification_level": "verified",
        "customer_address_stability": "stable",
        "customer_state": "CA",
        "product_category": "electronics",
        "product_subcategory": "smartphones",
        "product_brand": "Apple",
        "product_price": 299.99,
        "product_bnpl_eligible": True,
        "product_risk_category": "medium",
        "device_type": "mobile",
        "device_os": "iOS",
        "device_is_trusted": True,
        "payment_provider": "klarna",
        "payment_type": "bnpl_account",
        "payment_credit_limit": 1000,
        "_timestamp": "2025-09-26T14:30:00Z",
        "_record_id": 12345,
        "_generator": "BNPLGenerator"
    }

    # Add the transaction_timestamp field expected by our feature engineer
    # Map from _timestamp to transaction_timestamp
    real_api_transaction["transaction_timestamp"] = real_api_transaction["_timestamp"]

    print("🧪 Testing single transaction feature engineering with REAL API structure...")
    print(f"Input transaction: ${real_api_transaction['amount']} via {real_api_transaction['payment_provider']}")
    print(f"Customer: {real_api_transaction['customer_age_bracket']}, {real_api_transaction['customer_income_bracket']}")
    print(f"Device: {real_api_transaction['device_type']}, Category: {real_api_transaction['product_category']}")

    # Initialize feature engineer
    engineer = BNPLFeatureEngineer(client=None, verbose=True)

    try:
        # Process single transaction
        features_df = engineer.engineer_single_transaction(real_api_transaction)

        print(f"\n✅ Success! Generated features shape: {features_df.shape}")
        print(f"Expected: (1, 36)")

        # Show first few features
        print(f"\nFirst 10 features:")
        for i, (col, val) in enumerate(features_df.iloc[0].head(10).items()):
            print(f"  {i+1:2d}. {col}: {val}")

        print(f"\nLast 5 features:")
        for i, (col, val) in enumerate(features_df.iloc[0].tail(5).items(), start=32):
            print(f"  {i:2d}. {col}: {val}")

        # Check data types
        print(f"\nData types check:")
        print(f"  Numeric features: {features_df.select_dtypes(include=['number']).shape[1]}")
        print(f"  All features are numeric: {features_df.select_dtypes(include=['number']).shape[1] == 36}")

        # Show key one-hot encodings to verify they work
        print(f"\nKey one-hot encodings verification:")
        one_hot_features = ['device_type_mobile', 'device_type_tablet', 'payment_provider_klarna',
                           'product_category_electronics', 'purchase_context_normal']
        for feature in one_hot_features:
            if feature in features_df.columns:
                val = features_df[feature].iloc[0]
                print(f"  {feature}: {val}")

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_single_transaction()
    sys.exit(0 if success else 1)