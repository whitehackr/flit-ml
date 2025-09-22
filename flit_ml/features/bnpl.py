"""
BNPL-specific feature engineering utilities.

This module contains domain-specific feature engineering functions
for Buy Now, Pay Later risk assessment models.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class BNPLFeatureEngineer:
    """
    Feature engineering pipeline for BNPL risk assessment.

    Combines domain expertise with statistical feature engineering
    to create predictive features for default risk prediction.
    """

    def __init__(self):
        self.feature_config = {
            'velocity_features': True,
            'risk_indicators': True,
            'temporal_features': True,
            'customer_profile': True
        }

    def create_payment_velocity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create payment velocity and consistency features.

        Business Logic:
        - Faster payment = lower default risk
        - Consistent payment patterns = stable customer
        - Late payment history = red flag

        Args:
            df: Transaction-level data with payment information

        Returns:
            Customer-level payment velocity features
        """
        logger.info("Creating payment velocity features...")

        # Ensure datetime columns
        date_columns = ['transaction_date', 'payment_date', 'payment_due_date']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])

        # Calculate payment timing metrics
        if 'payment_date' in df.columns and 'transaction_date' in df.columns:
            df['days_to_payment'] = (df['payment_date'] - df['transaction_date']).dt.days

        if 'payment_due_date' in df.columns and 'payment_date' in df.columns:
            df['days_early_late'] = (df['payment_due_date'] - df['payment_date']).dt.days
            df['is_late_payment'] = (df['days_early_late'] < 0).astype(int)

        # Customer-level aggregations
        velocity_agg = {
            'transaction_id': 'count',  # Transaction frequency
        }

        # Add payment timing aggregations if available
        if 'days_to_payment' in df.columns:
            velocity_agg.update({
                'days_to_payment': ['mean', 'std', 'min', 'max']
            })

        if 'days_early_late' in df.columns:
            velocity_agg.update({
                'days_early_late': ['mean', 'std'],
                'is_late_payment': 'sum'
            })

        velocity_features = df.groupby('customer_id').agg(velocity_agg)

        # Flatten column names
        velocity_features.columns = [
            f"velocity_{col[0]}_{col[1]}" if isinstance(col, tuple)
            else f"velocity_{col}"
            for col in velocity_features.columns
        ]

        # Calculate derived metrics
        if 'velocity_is_late_payment_sum' in velocity_features.columns:
            velocity_features['velocity_late_payment_rate'] = (
                velocity_features['velocity_is_late_payment_sum'] /
                velocity_features['velocity_transaction_id_count']
            ).fillna(0).round(3)

        if 'velocity_days_to_payment_std' in velocity_features.columns:
            velocity_features['velocity_payment_consistency'] = (
                1 / (1 + velocity_features['velocity_days_to_payment_std'].fillna(0))
            ).round(3)

        logger.info(f"Created {velocity_features.shape[1]} velocity features for {velocity_features.shape[0]} customers")
        return velocity_features

    def create_risk_indicator_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create risk indicator features based on BNPL domain knowledge.

        Risk Factors:
        - High debt-to-income ratios
        - Frequent BNPL usage (over-borrowing)
        - Declining payment performance
        - Multiple failed payments

        Args:
            df: Transaction-level data with customer information

        Returns:
            Customer-level risk indicator features
        """
        logger.info("Creating risk indicator features...")

        # Basic risk aggregations
        risk_agg = {
            'transaction_amount': ['sum', 'mean', 'max', 'std', 'count'],
        }

        # Add customer profile aggregations if available
        profile_columns = ['credit_score', 'annual_income', 'customer_age']
        for col in profile_columns:
            if col in df.columns:
                risk_agg[col] = 'first'

        # Add failure indicators if available
        failure_columns = ['failed_payment_count', 'dispute_count', 'chargeback_count']
        for col in failure_columns:
            if col in df.columns:
                risk_agg[col] = 'sum'

        risk_features = df.groupby('customer_id').agg(risk_agg)

        # Flatten column names
        risk_features.columns = [
            f"risk_{col[0]}_{col[1]}" if isinstance(col, tuple)
            else f"risk_{col}"
            for col in risk_features.columns
        ]

        # Calculate derived risk metrics
        if 'risk_annual_income_first' in risk_features.columns:
            # Debt-to-income ratio
            risk_features['risk_debt_to_income_ratio'] = (
                risk_features['risk_transaction_amount_sum'] /
                risk_features['risk_annual_income_first']
            ).clip(0, 1).fillna(0).round(3)

            # Average transaction as % of monthly income
            risk_features['risk_transaction_to_monthly_income'] = (
                risk_features['risk_transaction_amount_mean'] /
                (risk_features['risk_annual_income_first'] / 12)
            ).clip(0, 2).fillna(0).round(3)

        # Transaction frequency risk
        risk_features['risk_monthly_transaction_frequency'] = (
            risk_features['risk_transaction_amount_count'] / 3  # Assume 3-month window
        ).round(2)

        # Transaction amount volatility
        if 'risk_transaction_amount_std' in risk_features.columns:
            risk_features['risk_spending_volatility'] = (
                risk_features['risk_transaction_amount_std'] /
                risk_features['risk_transaction_amount_mean'].replace(0, np.nan)
            ).fillna(0).round(3)

        logger.info(f"Created {risk_features.shape[1]} risk indicator features")
        return risk_features

    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract temporal patterns from transaction data.

        Temporal Insights:
        - Weekend vs weekday behavior
        - Month-end spending patterns
        - Seasonal trends
        - Time-of-day preferences

        Args:
            df: Transaction data with datetime information

        Returns:
            Customer-level temporal pattern features
        """
        logger.info("Creating temporal features...")

        # Ensure datetime conversion
        df['transaction_date'] = pd.to_datetime(df['transaction_date'])

        # Extract temporal components
        df['day_of_week'] = df['transaction_date'].dt.dayofweek
        df['day_of_month'] = df['transaction_date'].dt.day
        df['month'] = df['transaction_date'].dt.month
        df['hour'] = df['transaction_date'].dt.hour

        # Create binary indicators
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_month_end'] = (df['day_of_month'] > 25).astype(int)
        df['is_holiday_season'] = df['month'].isin([11, 12]).astype(int)
        df['is_business_hours'] = df['hour'].between(9, 17).astype(int)

        # Aggregate temporal patterns
        temporal_features = df.groupby('customer_id').agg({
            'is_weekend': 'mean',
            'is_month_end': 'mean',
            'is_holiday_season': 'mean',
            'is_business_hours': 'mean',
            'hour': ['mean', 'std'],
            'day_of_week': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 0
        }).round(3)

        # Flatten column names
        temporal_features.columns = [
            f"temporal_{col[0]}_{col[1]}" if isinstance(col, tuple)
            else f"temporal_{col}"
            for col in temporal_features.columns
        ]

        logger.info(f"Created {temporal_features.shape[1]} temporal features")
        return temporal_features

    def create_customer_profile_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create customer demographic and profile features.

        Args:
            df: Customer transaction data

        Returns:
            Customer-level profile features
        """
        logger.info("Creating customer profile features...")

        # Basic customer statistics
        profile_agg = {
            'transaction_amount': ['count', 'sum', 'mean', 'std', 'min', 'max'],
            'transaction_date': ['min', 'max']  # Customer tenure
        }

        # Add demographic features if available
        demographic_columns = [
            'customer_age', 'credit_score', 'annual_income',
            'employment_status', 'home_ownership_status'
        ]

        for col in demographic_columns:
            if col in df.columns:
                profile_agg[col] = 'first'

        profile_features = df.groupby('customer_id').agg(profile_agg)

        # Flatten column names
        profile_features.columns = [
            f"profile_{col[0]}_{col[1]}" if isinstance(col, tuple)
            else f"profile_{col}"
            for col in profile_features.columns
        ]

        # Calculate derived profile metrics
        if 'profile_transaction_date_min' in profile_features.columns:
            # Customer tenure in days
            profile_features['profile_customer_tenure_days'] = (
                pd.to_datetime(profile_features['profile_transaction_date_max']) -
                pd.to_datetime(profile_features['profile_transaction_date_min'])
            ).dt.days

        # Transaction patterns
        if 'profile_transaction_amount_std' in profile_features.columns:
            profile_features['profile_spending_consistency'] = (
                1 - (profile_features['profile_transaction_amount_std'] /
                     profile_features['profile_transaction_amount_mean'].replace(0, np.nan))
            ).fillna(0).clip(0, 1).round(3)

        logger.info(f"Created {profile_features.shape[1]} customer profile features")
        return profile_features

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Master feature engineering pipeline.

        Args:
            df: Raw transaction data

        Returns:
            Engineered features at customer level
        """
        logger.info("Starting BNPL feature engineering pipeline...")

        feature_sets = []

        # 1. Customer profile features
        if self.feature_config['customer_profile']:
            profile_features = self.create_customer_profile_features(df)
            feature_sets.append(profile_features)

        # 2. Payment velocity features
        if self.feature_config['velocity_features']:
            velocity_features = self.create_payment_velocity_features(df)
            feature_sets.append(velocity_features)

        # 3. Risk indicator features
        if self.feature_config['risk_indicators']:
            risk_features = self.create_risk_indicator_features(df)
            feature_sets.append(risk_features)

        # 4. Temporal features
        if self.feature_config['temporal_features']:
            temporal_features = self.create_temporal_features(df)
            feature_sets.append(temporal_features)

        # Combine all feature sets
        if feature_sets:
            final_features = pd.concat(feature_sets, axis=1)
        else:
            raise ValueError("No feature sets enabled in configuration")

        logger.info(f"Feature engineering complete: {final_features.shape[1]} features for {final_features.shape[0]} customers")

        return final_features

    def validate_features(self, features_df: pd.DataFrame) -> Dict:
        """
        Validate engineered features for production readiness.

        Args:
            features_df: Engineered features DataFrame

        Returns:
            Validation report dictionary
        """
        logger.info("Validating engineered features...")

        validation_report = {
            'feature_count': features_df.shape[1],
            'customer_count': features_df.shape[0],
            'missing_values': {},
            'constant_features': [],
            'high_correlation_pairs': [],
            'outlier_features': []
        }

        # Check for missing values
        missing_pct = (features_df.isnull().sum() / len(features_df) * 100).round(2)
        validation_report['missing_values'] = missing_pct[missing_pct > 0].to_dict()

        # Check for constant features (no variance)
        numeric_features = features_df.select_dtypes(include=[np.number])
        constant_features = numeric_features.columns[numeric_features.std() == 0].tolist()
        validation_report['constant_features'] = constant_features

        # Check for high correlations
        if len(numeric_features.columns) > 1:
            corr_matrix = numeric_features.corr().abs()
            high_corr_pairs = []

            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_value = corr_matrix.iloc[i, j]
                    if corr_value > 0.8:
                        high_corr_pairs.append({
                            'feature1': corr_matrix.columns[i],
                            'feature2': corr_matrix.columns[j],
                            'correlation': round(corr_value, 3)
                        })

            validation_report['high_correlation_pairs'] = high_corr_pairs

        # Check for outlier features (high skewness)
        outlier_features = []
        for col in numeric_features.columns:
            skewness = features_df[col].skew()
            if abs(skewness) > 5:
                outlier_features.append({
                    'feature': col,
                    'skewness': round(skewness, 2)
                })

        validation_report['outlier_features'] = outlier_features

        logger.info(f"Feature validation complete: {len(validation_report['missing_values'])} features with missing values, "
                   f"{len(constant_features)} constant features, {len(high_corr_pairs)} high correlation pairs")

        return validation_report


# Convenience functions for notebook usage
def engineer_bnpl_features(df: pd.DataFrame, config: Optional[Dict] = None) -> pd.DataFrame:
    """
    Convenience function for feature engineering in notebooks.

    Args:
        df: Raw transaction data
        config: Feature engineering configuration

    Returns:
        Engineered features DataFrame
    """
    engineer = BNPLFeatureEngineer()

    if config:
        engineer.feature_config.update(config)

    return engineer.engineer_features(df)


def validate_bnpl_features(features_df: pd.DataFrame) -> Dict:
    """
    Convenience function for feature validation in notebooks.

    Args:
        features_df: Engineered features DataFrame

    Returns:
        Validation report dictionary
    """
    engineer = BNPLFeatureEngineer()
    return engineer.validate_features(features_df)