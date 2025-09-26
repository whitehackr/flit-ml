# Production Artifacts - Multi-Model Research to Prod Transfer

## Multi-Model Production Strategy

**Deployment Mode:** Shadow deployment with champion/challenger setup
**Models:** 4 production candidates for simultaneous evaluation
**Rationale:** Similar performance (0.615-0.616 AUC) enables low-risk multi-model comparison

## What Must Be Extracted From Research

Based on the notebook analysis, here are the exact artifacts needed for multi-model production deployment:

### 1. The Trained Models (4 Models)
**Location:** `research/notebooks/03_model_development.ipynb`
**Models to Extract:**

```python
# Champion/Challenger Models
production_models = {
    'ridge': tuned_models['Ridge']['model'],           # 0.616 AUC, alpha=10.0
    'logistic': tuned_models['LogisticRegression']['model'],  # 0.615 AUC
    'elastic': tuned_models['ElasticNet']['model'],    # 0.615 AUC
    'ensemble': None  # Create voting ensemble from above 3
}
```

### 2. The Fitted Preprocessor (CRITICAL)
**Location:** Same notebook
**Variable:** `preprocessor`
**Type:** `ColumnTransformer` with fitted `StandardScaler`
**Structure:**
```python
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_features),    # FITTED scaler
    ('cat', 'passthrough', categorical_features)    # No transformation
])
```

**Why Critical:** This contains the **fitted scaling parameters** (mean/std) from training data that are essential for single-record inference.

### 3. Feature Names & Order
**Location:** Same notebook
**Variable:** `X_train.columns.tolist()`
**Purpose:** Ensure production features match training exactly

```python
production_feature_names = X_train.columns.tolist()  # 36 features
```

### 4. Feature Type Mapping
**Location:** Notebook preprocessing section
**Variables:** `numeric_features`, `categorical_features`
**Purpose:** Know which features get scaled vs passed through

```python
# Extract these lists from notebook
production_numeric_features = numeric_features
production_categorical_features = categorical_features
```

## Multi-Model Production Export Strategy

### Step 1: Create Multi-Model Export Function in Notebook
Add this cell to the end of the notebook:

```python
# MULTI-MODEL PRODUCTION EXPORT CELL
import joblib
import json
import os
from sklearn.ensemble import VotingClassifier

# Create models directory
os.makedirs('../../models/production', exist_ok=True)

# 1. Export all individual trained models
model_exports = {
    'ridge': tuned_models['Ridge']['model'],
    'logistic': tuned_models['LogisticRegression']['model'],
    'elastic': tuned_models['ElasticNet']['model']
}

for name, model in model_exports.items():
    joblib.dump(model, f'../../models/production/bnpl_{name}_v1.joblib')
    print(f"✅ Exported: bnpl_{name}_v1.joblib")

# 2. Create and export ensemble model
ensemble_model = VotingClassifier([
    ('ridge', model_exports['ridge']),
    ('logistic', model_exports['logistic']),
    ('elastic', model_exports['elastic'])
], voting='soft')

# Fit ensemble (required even though individual models are trained)
ensemble_model.fit(X_train_processed, y_train)
joblib.dump(ensemble_model, '../../models/production/bnpl_ensemble_v1.joblib')
print("✅ Exported: bnpl_ensemble_v1.joblib")

# 3. Export the fitted preprocessor (shared by all models)
joblib.dump(preprocessor, '../../models/production/bnpl_preprocessor_v1.joblib')
print("✅ Exported: bnpl_preprocessor_v1.joblib")

# 4. Export comprehensive metadata for all models
model_metadata = {
    'models': {
        'ridge': {
            'auc': 0.616,  # Update with actual values
            'parameters': tuned_models['Ridge']['model'].get_params(),
            'discrimination_ratio': 0.9
        },
        'logistic': {
            'auc': 0.615,  # Update with actual values
            'parameters': tuned_models['LogisticRegression']['model'].get_params(),
            'discrimination_ratio': 0.9
        },
        'elastic': {
            'auc': 0.615,  # Update with actual values
            'parameters': tuned_models['ElasticNet']['model'].get_params(),
            'discrimination_ratio': 0.9
        },
        'ensemble': {
            'type': 'VotingClassifier',
            'voting': 'soft',
            'estimators': ['ridge', 'logistic', 'elastic']
        }
    },
    'shared_artifacts': {
        'feature_names': X_train.columns.tolist(),
        'numeric_features': numeric_features,
        'categorical_features': categorical_features,
        'total_features': len(X_train.columns),
        'preprocessor': 'ColumnTransformer with StandardScaler'
    },
    'deployment_strategy': {
        'mode': 'shadow_multi_model',
        'champion': 'ridge',
        'challengers': ['logistic', 'elastic', 'ensemble'],
        'baseline_discrimination': 2.8,
        'target_improvement': 2.8
    }
}

with open('../../models/production/bnpl_multi_model_metadata_v1.json', 'w') as f:
    json.dump(model_metadata, f, indent=2)

print("✅ Production multi-model artifacts exported:")
print("   - 4 model files (ridge, logistic, elastic, ensemble)")
print("   - 1 shared preprocessor")
print("   - 1 comprehensive metadata file")
```

### Step 2: Multi-Model Production Loading Pattern
```python
# In production code:
class BNPLMultiModelPredictor:
    def __init__(self):
        # Load all models
        self.models = {
            'ridge': joblib.load('models/production/bnpl_ridge_v1.joblib'),
            'logistic': joblib.load('models/production/bnpl_logistic_v1.joblib'),
            'elastic': joblib.load('models/production/bnpl_elastic_v1.joblib'),
            'ensemble': joblib.load('models/production/bnpl_ensemble_v1.joblib')
        }

        # Shared preprocessor for all models
        self.preprocessor = joblib.load('models/production/bnpl_preprocessor_v1.joblib')

        # Load metadata
        with open('models/production/bnpl_multi_model_metadata_v1.json', 'r') as f:
            self.metadata = json.load(f)

        self.feature_names = self.metadata['shared_artifacts']['feature_names']
        self.champion = self.metadata['deployment_strategy']['champion']

    def predict_single_transaction(self, transaction_features):
        """
        Get predictions from all models for shadow mode comparison

        Returns:
        {
            'ridge': 0.23,
            'logistic': 0.25,
            'elastic': 0.24,
            'ensemble': 0.24,
            'champion': 'ridge',
            'champion_score': 0.23,
            'consensus': 0.24  # average of all models
        }
        """
        # Preprocess once, use for all models
        X_scaled = self.preprocessor.transform(transaction_features)

        predictions = {}
        for name, model in self.models.items():
            predictions[name] = float(model.predict_proba(X_scaled)[0][1])

        # Add meta-predictions
        predictions['champion'] = self.champion
        predictions['champion_score'] = predictions[self.champion]
        predictions['consensus'] = sum(predictions.values()) / len(self.models)

        return predictions

    def get_champion_decision(self, transaction_features):
        """Production decision using champion model only"""
        predictions = self.predict_single_transaction(transaction_features)
        return predictions['champion_score']
```

## Single Transaction Processing Architecture

The key insight: **The preprocessor contains the fitted scaling parameters**, so we can process single records using the same parameters that were fitted on the training batch.

### Current Gap: Feature Engineering for Single Records
`BNPLFeatureEngineer` currently assumes batch processing from BigQuery. We need to add:

```python
# Addition to BNPLFeatureEngineer class:
def engineer_single_transaction(self, transaction_data: dict) -> pd.DataFrame:
    """
    Transform single transaction into the 36 features expected by model.

    Input: Raw transaction dict from API
    Output: DataFrame with 1 row, 36 columns matching training feature order
    """
```

## Shadow Deployment Architecture

### Multi-Model Shadow Mode Implementation
```python
class ShadowModeController:
    def __init__(self):
        self.multi_model = BNPLMultiModelPredictor()
        self.business_rules = ExistingBusinessRules()  # Current system

    async def process_transaction_request(self, transaction_data):
        # 1. Get predictions from all ML models (shadow mode - no business impact)
        ml_predictions = self.multi_model.predict_single_transaction(transaction_data)

        # 2. Get existing business rules decision (this is what actually gets used)
        business_decision = self.business_rules.evaluate(transaction_data)

        # 3. Log all predictions for analysis (no impact on actual decision)
        await self.log_shadow_predictions({
            'transaction_id': transaction_data['transaction_id'],
            'business_decision': business_decision,
            'ml_predictions': ml_predictions,
            'timestamp': datetime.utcnow()
        })

        # 4. Return business rules decision (ML models are purely observational)
        return business_decision

    async def log_shadow_predictions(self, prediction_data):
        """Log to database for dashboard and performance analysis"""
        # Store in predictions table for Streamlit dashboard
        # Include all model predictions for comparison
```

### Benefits of Multi-Model Shadow Deployment
1. **Risk-Free Comparison:** Evaluate 4 models simultaneously without business impact
2. **Rich Data Collection:** Compare Ridge vs LogisticRegression vs ElasticNet vs Ensemble
3. **Performance Baseline:** Measure actual discrimination ratios vs 2.8x business rules
4. **Champion Selection:** Data-driven decision on which model to activate
5. **Feature Issue Detection:** Monitor if feature importance fixes improve any model significantly

## Dashboard & Monitoring Artifacts

For the Streamlit dashboard, we'll need to track all models:

### 4. Training Data Statistics (for drift detection)
```python
# Export training data statistics for monitoring
training_stats = {
    'feature_distributions': {
        feature: {
            'mean': float(X_train[feature].mean()),
            'std': float(X_train[feature].std()),
            'min': float(X_train[feature].min()),
            'max': float(X_train[feature].max())
        } for feature in X_train.columns
    },
    'target_distribution': {
        'default_rate': float(y_train.mean()),
        'total_samples': len(y_train)
    }
}
```

### 5. Multi-Model Performance Baseline Metrics
```python
baseline_metrics = {
    'business_rules': {
        'discrimination_ratio': 2.8,
        'high_risk_default_rate': 0.09,  # 9%
        'low_risk_default_rate': 0.032   # 3.2%
    },
    'ml_models': {
        'ridge': {
            'auc': 0.616,
            'discrimination_ratio': 0.9  # Update from notebook
        },
        'logistic': {
            'auc': 0.615,
            'discrimination_ratio': 0.9  # Update from notebook
        },
        'elastic': {
            'auc': 0.615,
            'discrimination_ratio': 0.9  # Update from notebook
        },
        'ensemble': {
            'expected_auc': 0.616,  # Similar or slightly better
            'discrimination_ratio': 'TBD'  # Measure in shadow mode
        }
    },
    'targets': {
        'minimum_improvement': 1.0,   # Better than random
        'success_threshold': 2.8,    # Beat business rules
        'acceptable_range': [1.5, 2.8]  # Partial improvement acceptable
    }
}
```

### 6. Streamlit Dashboard Schema
```python
# Database schema for dashboard tracking
shadow_predictions_table = {
    'transaction_id': 'string',
    'timestamp': 'datetime',
    'business_decision': 'string',  # 'approve' or 'decline'
    'business_risk_score': 'float',
    'ridge_prediction': 'float',
    'logistic_prediction': 'float',
    'elastic_prediction': 'float',
    'ensemble_prediction': 'float',
    'consensus_prediction': 'float',
    'actual_default': 'boolean',  # Updated after 2+ weeks
    'days_to_resolution': 'integer',  # Days until we know outcome
    'opportunity_cost': 'float'  # Business impact if ML decision was used
}
```

## Next Steps

### Phase 1: Research Completion
1. **Add multi-model export cell to notebook** and run it
2. **Extract all 4 models** (Ridge, LogisticRegression, ElasticNet, Ensemble)
3. **Validate exported artifacts** work correctly
4. **Commit production artifacts** to research branch

### Phase 2: Production Pipeline Development (Separate PR)
1. **Modify BNPLFeatureEngineer** to handle single transactions
2. **Build BNPLMultiModelPredictor** class
3. **Implement ShadowModeController**
4. **Create API endpoints** with multi-model support
5. **Build Streamlit dashboard** for model comparison

### Phase 3: Shadow Deployment
1. **Deploy multi-model shadow mode** to production
2. **Collect 4-6 weeks of shadow predictions** vs business rules
3. **Monitor actual defaults** vs all model predictions
4. **Select champion model** based on real performance data
5. **Activate best-performing model** for live decisions

## Multi-Model Benefits Summary

- **4 models running simultaneously** in shadow mode
- **Risk-free evaluation** of all approaches
- **Rich comparison data** for champion selection
- **Ensemble learning** potential with voting classifier
- **Feature engineering validation** across multiple algorithms
- **Business impact measurement** via opportunity cost analysis

Ready to proceed with multi-model export from notebook!