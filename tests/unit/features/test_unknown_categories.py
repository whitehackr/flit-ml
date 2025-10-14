#!/usr/bin/env python3
"""
Test single transaction feature engineering with unknown categories
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from flit_ml.features.bnpl_feature_engineering import BNPLFeatureEngineer

def test_unknown_categories():
    """Test handling of unknown category values."""

    # Transaction with unknown categories (like "affirm" from real BigQuery data)
    transaction_with_unknowns = {
        "transaction_id": "txn_test_002",
        "customer_id": "cust_test_002",
        "amount": 94.59,
        "transaction_timestamp": "2025-02-18T00:00:02Z",
        "customer_credit_score_range": "fair",
        "customer_age_bracket": "25-34",
        "customer_income_bracket": "25k-50k",
        "customer_verification_level": "unverified",
        "customer_tenure_days": 264,
        "device_type": "mobile",
        "device_is_trusted": True,
        "product_category": "home",
        "product_risk_category": "medium",
        "risk_score": 0.6,
        "risk_level": "high",
        "risk_scenario": "high_risk_behavior",
        "payment_provider": "affirm",  # NOT in training data (afterpay, klarna, sezzle)
        "installment_count": 4,
        "payment_credit_limit": 500,
        "price_comparison_time": 3,
        "purchase_context": "rushed"
    }

    print("🧪 Testing unknown categories handling...")
    print(f"Payment provider: {transaction_with_unknowns['payment_provider']} (unknown)")
    print(f"Expected: All payment_provider_* features should be 0")

    # Initialize feature engineer
    engineer = BNPLFeatureEngineer(client=None, verbose=False)

    try:
        # Process transaction
        features_df = engineer.engineer_single_transaction(transaction_with_unknowns)

        print(f"\n✅ Success! Shape: {features_df.shape}")

        # Check payment provider one-hot encoding
        payment_features = [
            'payment_provider_afterpay',
            'payment_provider_klarna',
            'payment_provider_sezzle'
        ]

        print(f"\nPayment provider one-hot encoding (should all be 0):")
        all_zero = True
        for feature in payment_features:
            if feature in features_df.columns:
                val = features_df[feature].iloc[0]
                print(f"  {feature}: {val}")
                if val != 0:
                    all_zero = False

        print(f"\n✅ Unknown category handling: {'PASS' if all_zero else 'FAIL'}")

        # Also test other features work correctly
        print(f"\nOther features still work correctly:")
        print(f"  device_type_mobile: {features_df['device_type_mobile'].iloc[0]} (should be 1)")
        print(f"  product_category_home: {features_df['product_category_home'].iloc[0]} (should be 1)")
        print(f"  purchase_context_rushed: {features_df['purchase_context_rushed'].iloc[0]} (should be 1)")

        return all_zero

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_unknown_categories()
    print(f"\n{'🎉 All tests passed!' if success else '❌ Tests failed!'}")
    sys.exit(0 if success else 1)