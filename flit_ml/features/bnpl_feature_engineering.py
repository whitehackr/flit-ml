"""
BNPL Feature Engineering Module

Complete feature engineering pipeline for Buy Now Pay Later (BNPL) default risk prediction.
Handles data loading, temporal features, categorical encoding, cleaning, and optimization
for production deployment with <100ms inference latency.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Tuple, Optional
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from google.cloud import bigquery
from flit_ml.config import config


class BNPLFeatureEngineer:
    """
    Complete feature engineering pipeline for BNPL default risk prediction.

    Handles data loading from BigQuery, temporal feature extraction, categorical encoding,
    data cleaning, and feature optimization for production deployment.
    """

    def __init__(self, client: Optional[bigquery.Client] = None, verbose: bool = True, log_level: int = logging.INFO):
        """
        Initialize BNPL Feature Engineer.

        Args:
            client: BigQuery client. If None, creates from config.
            verbose: If True, prints clean progress messages to console.
                    If False, no console output at all.
            log_level: Logging level (logging.DEBUG, logging.INFO, etc.)
        """
        self.client = client or config.get_client()
        self.verbose = verbose
        self.feature_metadata = {}

        # Setup logger (always logs to file, never to console)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)

        # Setup file logging handler if not already configured
        if not self.logger.handlers:
            # Create logs directory at project root if it doesn't exist
            import os
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            log_dir = os.path.join(project_root, 'logs')
            os.makedirs(log_dir, exist_ok=True)

            # File handler - always logs to file
            file_handler = logging.FileHandler(os.path.join(log_dir, 'bnpl_feature_engineering.log'))
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    def _log_and_print(self, message: str, level: int = logging.INFO):
        """
        Log message to file (always) and optionally print to console based on verbose setting.

        Args:
            message: Message to log/print
            level: Logging level (logging.INFO, logging.DEBUG, etc.)
        """
        # Always log to file
        self.logger.log(level, message)

        # Only print to console if verbose mode enabled
        if self.verbose:
            print(message)

    def load_bnpl_data(self, sample_size: Optional[int] = None, random_seed: int = 42) -> pd.DataFrame:
        """
        Load BNPL transaction data from BigQuery with configurable sample size.

        Args:
            sample_size: Number of customers to sample. If None, loads all data.
            random_seed: Random seed for reproducible sampling.

        Returns:
            DataFrame with BNPL transaction data
        """

        # Build customer sampling query
        if sample_size:
            customer_sample_query = f"""
            WITH customer_sample AS (
              SELECT DISTINCT customer_id
              FROM `flit-data-platform.flit_intermediate.int_bnpl_customer_tenure_adjusted`
              ORDER BY FARM_FINGERPRINT(CONCAT(customer_id, '{random_seed}'))
              LIMIT {sample_size}
            )
            """
            join_clause = "INNER JOIN customer_sample cs ON t.customer_id = cs.customer_id"
        else:
            customer_sample_query = ""
            join_clause = ""

        # Main data query
        feature_data_query = f"""
        {customer_sample_query}
        SELECT
            -- t.customer_id, -- should work in prod
            t.unique_customer_id as customer_id, -- in dev, when data is from BQ
            t.transaction_id,
            t.amount,
            t.will_default,
            t.transaction_timestamp,
            t.days_to_first_missed_payment,

            -- Customer features (available at transaction time)
            t.customer_credit_score_range,
            t.customer_age_bracket,
            t.customer_income_bracket,
            t.customer_verification_level,
            --t.customer_tenure_days, -- for prod
            t.adjusted_customer_tenure as customer_tenure_days, -- in dev
            t.customer_state,

            -- Transaction context features
            t.product_category,
            t.product_risk_category,
            t.product_price,
            t.device_type,
            t.device_is_trusted,

            -- Current underwriting features (baseline to beat)
            t.risk_score,
            t.risk_level,
            t.risk_scenario,

            -- Additional transaction context
            t.payment_provider,
            t.installment_count,
            t.payment_credit_limit,
            t.payment_type,
            t.time_on_site_seconds,
            t.purchase_context,
            t.price_comparison_time

        FROM `flit-data-platform.flit_intermediate.int_bnpl_customer_tenure_adjusted` t
        {join_clause}
        ORDER BY t.customer_id, t.transaction_timestamp
        """

        self._log_and_print(f"📥 Loading BNPL data from BigQuery...")
        if sample_size:
            self._log_and_print(f"   Sample size: {sample_size:,} customers")
        else:
            self._log_and_print(f"   Loading all available data")

        df = self.client.query(feature_data_query).to_dataframe()

        self._log_and_print(f"✅ Data loaded: {df.shape[0]:,} transactions for {df['customer_id'].nunique():,} customers")
        self._log_and_print(f"   Default rate: {df['will_default'].mean():.1%}")
        self._log_and_print(f"   Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

        return df

    def extract_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract temporal features from transaction_timestamp.

        Business Logic:
        - Hour patterns: Late night/early morning may indicate impulsive behavior
        - Day patterns: Weekend vs weekday spending behaviors
        - Month patterns: Holiday seasons, month-end financial stress

        Args:
            df: DataFrame with transaction_timestamp column

        Returns:
            DataFrame with additional temporal features
        """
        df = df.copy()

        # Ensure transaction_timestamp is datetime
        df['transaction_timestamp'] = pd.to_datetime(df['transaction_timestamp'])

        # Extract basic temporal components
        df['transaction_hour'] = df['transaction_timestamp'].dt.hour
        df['transaction_day_of_week'] = df['transaction_timestamp'].dt.dayofweek  # 0=Monday, 6=Sunday
        df['transaction_month'] = df['transaction_timestamp'].dt.month
        df['transaction_day_of_month'] = df['transaction_timestamp'].dt.day

        # Week of month calculation
        df['week_of_month'] = ((df['transaction_day_of_month'] - 1) // 7) + 1

        # Create categorical temporal features
        df['is_weekend'] = df['transaction_day_of_week'].isin([5, 6]).astype(int)  # Saturday, Sunday
        df['is_month_end'] = (df['transaction_day_of_month'] >= 25).astype(int)  # Last week of month
        df['is_holiday_season'] = df['transaction_month'].isin([11, 12]).astype(int)  # Nov, Dec
        df['is_business_hours'] = df['transaction_hour'].between(9, 17).astype(int)  # 9 AM - 5 PM
        df['is_late_night'] = df['transaction_hour'].isin([22, 23, 0, 1, 2, 3, 4, 5]).astype(int)  # 10 PM - 5 AM

        # Time of day categories
        df['time_of_day'] = pd.cut(
            df['transaction_hour'],
            bins=[-1, 6, 12, 18, 24],
            labels=['night', 'morning', 'afternoon', 'evening']
        )

        self._log_and_print(f"✅ Temporal features extracted:")
        temporal_cols = [col for col in df.columns if col.startswith(('transaction_', 'is_', 'time_of_day', 'week_of_month'))]
        for col in temporal_cols:
            unique_vals = df[col].nunique()
            self._log_and_print(f"   {col}: {unique_vals} unique values")

        return df

    def encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical variables using appropriate sklearn encoders.

        Strategy:
        - Ordinal encoding for ordered categories (preserves ranking)
        - One-hot encoding for nominal categories (no inherent order)

        Args:
            df: DataFrame with categorical features

        Returns:
            DataFrame with encoded categorical features
        """
        df_encoded = df.copy()

        # Define ordinal mappings (order matters!)
        ordinal_mappings = {
            'customer_credit_score_range': ['poor', 'fair', 'good', 'excellent'],
            'customer_age_bracket': ['18-24', '25-34', '35-44', '45-54', '55+'],
            'customer_income_bracket': ['<25k', '25k-50k', '50k-75k', '75k-100k', '100k+'],
            'customer_verification_level': ['unverified', 'partial', 'verified'],
            'product_risk_category': ['low', 'medium', 'high'],
            'risk_level': ['low', 'medium', 'high']
        }

        # Apply ordinal encoding
        self._log_and_print("🔢 Applying ordinal encoding...")
        for feature, categories in ordinal_mappings.items():
            if feature in df_encoded.columns:
                # Check if all expected categories exist
                actual_categories = df_encoded[feature].unique()
                self._log_and_print(f"   {feature}: {actual_categories}")

                # Create ordinal encoder
                ordinal_encoder = OrdinalEncoder(
                    categories=[categories],
                    handle_unknown='use_encoded_value',
                    unknown_value=-1
                )

                # Apply encoding
                encoded_values = ordinal_encoder.fit_transform(df_encoded[[feature]])
                df_encoded[f"{feature}_encoded"] = encoded_values.flatten()

        # Define nominal features for one-hot encoding
        nominal_features = [
            'device_type', 'payment_provider', 'product_category',
            'purchase_context', 'risk_scenario', 'time_of_day'
        ]

        # Check cardinality before one-hot encoding
        self._log_and_print(f"\n🏷️  Checking cardinality for one-hot encoding...")
        for feature in nominal_features:
            if feature in df_encoded.columns:
                cardinality = df_encoded[feature].nunique()
                unique_vals = df_encoded[feature].unique()
                self._log_and_print(f"   {feature}: {cardinality} unique values {unique_vals}")

        # Apply one-hot encoding using sklearn OneHotEncoder
        self._log_and_print(f"\n🎯 Applying one-hot encoding...")
        for feature in nominal_features:
            if feature in df_encoded.columns:
                # Create OneHot encoder
                onehot_encoder = OneHotEncoder(
                    sparse_output=False,
                    drop='first',
                    handle_unknown='ignore'
                )

                # Fit and transform
                encoded_values = onehot_encoder.fit_transform(df_encoded[[feature]])

                # Get feature names
                feature_names = onehot_encoder.get_feature_names_out([feature])

                # Add encoded columns to dataframe
                encoded_df = pd.DataFrame(encoded_values, columns=feature_names, index=df_encoded.index)
                df_encoded = pd.concat([df_encoded, encoded_df], axis=1)

                self._log_and_print(f"   {feature}: Created {len(feature_names)} dummy variables")

        # Boolean features are already encoded (device_is_trusted: True/False → 1/0)
        self._log_and_print(f"\n✅ Boolean features already encoded:")
        boolean_features = df_encoded.select_dtypes(include=['bool']).columns
        for feature in boolean_features:
            self._log_and_print(f"   {feature}: {df_encoded[feature].dtype}")
        
        # Explicitly convert boolean columns to int (0/1)
        df_encoded['will_default'] = df_encoded['will_default'].astype(int)
        df_encoded['device_is_trusted'] = df_encoded['device_is_trusted'].astype(int)

        self._log_and_print(f"\n📊 Encoding Summary:")
        original_categorical = len([col for col in df.columns if df[col].dtype == 'object'])
        new_encoded_features = len([col for col in df_encoded.columns if col.endswith('_encoded') or '_' in col and col not in df.columns])
        self._log_and_print(f"   Original categorical features: {original_categorical}")
        self._log_and_print(f"   New encoded features created: {new_encoded_features}")
        self._log_and_print(f"   Total features: {len(df_encoded.columns)}")

        return df_encoded

    def clean_and_select_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Clean data and select optimal feature set for production deployment.

        Strategy:
        1. Remove redundant features (original categorical after encoding)
        2. Drop identifier fields (not predictive)
        3. Remove low-value temporal components (keep engineered features)
        4. Handle problematic fields (data leakage, high cardinality)
        5. Address missing values appropriately

        Args:
            df: DataFrame with encoded features

        Returns:
            Tuple of (cleaned_dataframe, feature_metadata)
        """
        df_clean = df.copy()

        self._log_and_print("🧹 Starting data cleaning and feature selection...")
        self._log_and_print(f"   Starting features: {len(df_clean.columns)}")

        # 1. Remove redundant original categorical features (keep encoded versions)
        redundant_categorical = [
            'customer_credit_score_range',  # Keep: customer_credit_score_range_encoded
            'customer_age_bracket',         # Keep: customer_age_bracket_encoded
            'customer_income_bracket',      # Keep: customer_income_bracket_encoded
            'customer_verification_level',  # Keep: customer_verification_level_encoded
            'product_risk_category',        # Keep: product_risk_category_encoded
            'risk_level',                   # Keep: risk_level_encoded
            'device_type',                  # Keep: device_type_* dummies
            'payment_provider',             # Keep: payment_provider_* dummies
            'product_category',             # Keep: product_category_* dummies
            'purchase_context',             # Keep: purchase_context_* dummies
            'risk_scenario',                # Keep: risk_scenario_* dummies
            'time_of_day'                   # Keep: time_of_day_* dummies
        ]

        # 2. Remove identifier fields (not predictive)
        identifier_fields = [
            'customer_id',                  # Just identifier
            'transaction_id',                # Just identifier
            'record_id',
            'unique_transaction_id',
            'unique_customer_id',
            'first_customer_transaction_date',
            'first_customer_tenure',
            'days_from_first_transaction',
            'device_id',
            'product_id',
            'transaction_timestamp',  # Keep derived temporal features, drop raw
            'ingestion_timestamp',  # Metadata, not predictive
        ]

        # 3. Remove raw temporal components (keep engineered features)
        raw_temporal = [
            'transaction_timestamp',        # Keep derived temporal features
            'transaction_hour',             # Keep: is_late_night, is_business_hours
            'transaction_day_of_week',      # Keep: is_weekend
            'transaction_month',            # Keep: is_holiday_season
            'transaction_day_of_month'      # Keep: is_month_end, week_of_month
        ]

        # 4. Handle problematic fields
        problematic_fields = []

        # Check for data leakage indicators
        # if 'days_to_first_missed_payment' in df_clean.columns:
        #     unique_vals = df_clean['days_to_first_missed_payment'].unique()
        #     print(f"   ⚠️  days_to_first_missed_payment values: {unique_vals}")
        #     if len(unique_vals) > 1:  # If not all the same, might be leakage
        #         print(f"      → Potential data leakage detected. Consider removing.")
        #         problematic_fields.append('days_to_first_missed_payment')

        # Check for single-value features (no predictive power)
        single_value_features = []
        for col in df_clean.columns:
            if df_clean[col].nunique() <= 1:
                single_value_features.append(col)

        if single_value_features:
            self._log_and_print(f"   🗑️  Single-value features to remove: {single_value_features}")
            problematic_fields.extend(single_value_features)

        # Check high cardinality categorical features
        high_cardinality_features = []
        categorical_cols = df_clean.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            cardinality = df_clean[col].nunique()
            if cardinality > 20:  # Threshold for high cardinality
                high_cardinality_features.append(col)
                self._log_and_print(f"   📊 High cardinality feature: {col} ({cardinality} values)")

        # 5. Plain unnecessary features or alraedy adjusted fields:
        unnecessary_fields = [
            '_generator',
            '_timestamp',
            'currency',
            'customer_state',
            #'customer_tenure_days',
            'device_os',
            'payment_frequency',        # removing it for now, because we only have biweekly. Would be critical when we intro new terms
            'payment_method_id',
            'payment_type',
            'product_price',             # Not needed as we have amount
            'status',
            'json_body',
            'data_source',
            '_loaded_at',
            ]

        # Combine all features to drop
        features_to_drop = (redundant_categorical + identifier_fields +
                           raw_temporal + problematic_fields + high_cardinality_features + unnecessary_fields)

        # Remove duplicates and features that don't exist
        features_to_drop = list(set(features_to_drop))
        existing_features_to_drop = [f for f in features_to_drop if f in df_clean.columns]

        self._log_and_print(f"\n🗑️  Removing {len(existing_features_to_drop)} features:")
        for feature in existing_features_to_drop:
            self._log_and_print(f"   - {feature}")

        # Drop the features
        df_clean = df_clean.drop(columns=existing_features_to_drop)

        # 5. Handle missing values
        self._log_and_print(f"\n🔍 Checking for missing values...")
        missing_summary = df_clean.isnull().sum()
        missing_features = missing_summary[missing_summary > 0]

        if len(missing_features) > 0:
            self._log_and_print(f"   Missing values found:")
            for feature, count in missing_features.items():
                pct = (count / len(df_clean)) * 100
                self._log_and_print(f"   - {feature}: {count:,} ({pct:.1f}%)")

                # Handle missing values based on data type
                if df_clean[feature].dtype in ['float64', 'int64']:
                    # Numeric: fill with median
                    median_val = df_clean[feature].median()
                    df_clean[feature].fillna(median_val, inplace=True)
                    self._log_and_print(f"     → Filled with median: {median_val}")
                else:
                    # Categorical: fill with mode or 'unknown'
                    if df_clean[feature].mode().empty:
                        df_clean[feature].fillna('unknown', inplace=True)
                        self._log_and_print(f"     → Filled with: 'unknown'")
                    else:
                        mode_val = df_clean[feature].mode().iloc[0]
                        df_clean[feature].fillna(mode_val, inplace=True)
                        self._log_and_print(f"     → Filled with mode: {mode_val}")
        else:
            self._log_and_print(f"   ✅ No missing values found")

        # 6. Convert string numeric fields to proper numeric types
        string_numeric_fields = ['time_on_site_seconds', 'price_comparison_time']
        for field in string_numeric_fields:
            if field in df_clean.columns and df_clean[field].dtype == 'object':
                df_clean[field] = pd.to_numeric(df_clean[field], errors='coerce')
                self._log_and_print(f"   🔢 Converted {field} to numeric")

        # 7. Final feature summary and metadata
        self._log_and_print(f"\n📊 Final Feature Set Summary:")
        self._log_and_print(f"   Final features: {len(df_clean.columns)}")
        self._log_and_print(f"   Features removed: {len(existing_features_to_drop)}")
        self._log_and_print(f"   Data shape: {df_clean.shape}")

        # Categorize remaining features
        numeric_features = df_clean.select_dtypes(include=[np.number]).columns.tolist()
        boolean_features = df_clean.select_dtypes(include=['bool']).columns.tolist()
        categorical_features = df_clean.select_dtypes(include=['object']).columns.tolist()

        self._log_and_print(f"\n📋 Feature Types:")
        self._log_and_print(f"   Numeric features: {len(numeric_features)}")
        self._log_and_print(f"   Boolean features: {len(boolean_features)}")
        self._log_and_print(f"   Categorical features: {len(categorical_features)}")

        if categorical_features:
            self._log_and_print(f"   ⚠️  Remaining categorical features may need encoding: {categorical_features}")

        # Save feature metadata for production deployment
        feature_metadata = {
            'all_features': df_clean.columns.tolist(),
            'numeric_features': numeric_features,
            'boolean_features': boolean_features,
            'categorical_features': categorical_features,
            'target_variable': 'will_default',
            'features_removed': existing_features_to_drop,
            'data_shape': df_clean.shape,
            'sample_info': {
                'total_transactions': len(df_clean),
                'unique_customers': df_clean.get('customer_id', pd.Series()).nunique() if 'customer_id' in df.columns else 'unknown',
                'default_rate': df_clean['will_default'].mean() if 'will_default' in df_clean.columns else 'unknown'
            }
        }

        self.feature_metadata = feature_metadata

        return df_clean, feature_metadata

    def engineer_features(self, sample_size: Optional[int] = None, random_seed: int = 42) -> Tuple[pd.DataFrame, Dict]:
        """
        Complete end-to-end feature engineering pipeline.

        Orchestrates the full pipeline: data loading → temporal features →
        categorical encoding → data cleaning and feature selection.

        Args:
            sample_size: Number of customers to sample. If None, loads all data.
            random_seed: Random seed for reproducible sampling.

        Returns:
            Tuple of (final_dataframe, feature_metadata)
        """
        self._log_and_print("🚀 Starting complete BNPL feature engineering pipeline...")

        # Step 1: Load data
        df = self.load_bnpl_data(sample_size=sample_size, random_seed=random_seed)

        # Step 2: Extract temporal features
        self._log_and_print(f"\n⏰ Step 2: Extracting temporal features...")
        df = self.extract_temporal_features(df)

        # Step 3: Encode categorical features
        self._log_and_print(f"\n🔧 Step 3: Encoding categorical features...")
        df = self.encode_categorical_features(df)

        # Step 4: Clean and select features
        self._log_and_print(f"\n🧹 Step 4: Cleaning and selecting features...")
        df_final, feature_metadata = self.clean_and_select_features(df)

        self._log_and_print(f"\n✅ Feature engineering pipeline complete!")
        self._log_and_print(f"   Final dataset shape: {df_final.shape}")
        self._log_and_print(f"   Ready for ML model development")

        return df_final, feature_metadata