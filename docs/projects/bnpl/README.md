# BNPL Default Risk Prediction

Production ML system for real-time Buy Now Pay Later credit decisioning with shadow mode A/B testing infrastructure.

## Project Overview

**Objective**: Real-time credit decisioning for BNPL transactions with <100ms latency
**Deployment**: Railway (production) with Redis caching and MLflow tracking
**Architecture**: Shadow mode controller running 4 models in parallel for safe production comparison
**Data Source**: Simtom synthetic eCommerce transactions (1.9M training records)
**Status**: Phase 2 complete - deployed with ephemeral MLflow, Phase 3 planned for persistent infrastructure

## Business Problem

Buy Now Pay Later products require millisecond-speed credit decisions during checkout using only transaction-time data, without the payment histories or credit bureau depth available to traditional underwriting. The business operates a rule-based risk system achieving 2.8x discrimination ratio (high-risk defaults at 9% versus low-risk at 3.2%). Any ML system must integrate with existing underwriting workflows while maintaining sub-100ms latency to avoid degrading checkout conversion rates.

The technical challenge extends beyond model accuracy to encompass experiment management for safe model updates, comprehensive performance tracking to detect drift in synthetic data distributions, and business rule integration that maintains risk threshold control as market conditions evolve.

## Technical Architecture

### System Flow

The production system separates three distinct concerns: **prediction generation** (pure ML inference), **deployment strategy** (A/B testing and traffic allocation), and **business integration** (risk thresholds and decision policies). This architectural separation enables independent optimization of each layer.

**Inference Flow**:
1. Simtom sends transaction JSON → ML API receives request
2. Shadow Controller engineers 36 features in-memory (~15ms)
3. ALL 4 models generate predictions in parallel (~2ms total)
4. Experiment logic selects which model's prediction to use (A/B testing)
5. Business decision engine applies risk thresholds → Decision returned to Simtom
6. **Async (non-blocking)**: ALL 4 predictions + input data cached to Redis
7. **Async**: Experiment data logged to MLflow
8. **Daily batch**: Redis → BigQuery for long-term storage and retraining

**Key Design Pattern**: Synchronous path (<20ms) handles only essential operations for business decision. All logging, caching, and analytics happen asynchronously to maintain low latency.

### Production Components

**Multi-Model Predictor** (`flit_ml/models/bnpl/predictor.py:20-222`)
- Loads 4 production models: Ridge (champion), Logistic Regression, Elastic Net, Voting Ensemble
- Supports flexible deployment modes: shadow (all models), champion-only, specific model selection
- Shared preprocessor with fitted StandardScaler ensures training/inference consistency
- Sub-millisecond inference per model enables parallel execution without latency impact

**Shadow Controller** (`flit_ml/core/shadow_controller.py:326-797`)
- Orchestrates complete prediction workflow with experiment management
- Deterministic traffic allocation via customer ID hashing (consistent assignment across transactions)
- Configurable decision policies: Conservative/Balanced/Aggressive risk thresholds
- Storage abstraction: in-memory (dev) → Redis (production) → future enhanced storage

**Feature Engineering** (`flit_ml/features/bnpl_feature_engineering.py:553-829`)
- Single-transaction pipeline generates 36 features from transaction JSON
- Temporal extraction: hour patterns, weekend indicators, holiday season flags
- Categorical encoding: ordinal for ranked features (credit scores), one-hot for nominals (product categories)
- Unknown category handling: graceful degradation for values absent from training data
- Processing time: ~15ms maintains <100ms total latency budget

## Key Design Decisions

### Model Selection: Ridge Classifier

**Problem**: Balance predictive performance, inference speed, and interpretability for production deployment.

**Evaluation**: Tested 20+ algorithms including tree ensembles (XGBoost, LightGBM), linear models, neural networks, and custom ensembles. Ridge Classifier selected as production candidate based on:
- **Speed**: Sub-millisecond inference enables 4-model parallel execution
- **Simplicity**: Linear architecture reduces deployment complexity
- **Interpretability**: Coefficient inspection enables business stakeholder communication
- **Scalability**: Minimal memory footprint supports horizontal scaling

**Trade-off**: Ensemble methods achieved marginally better cross-validation metrics but at 16+ seconds training time and 100ms+ inference latency, unacceptable for real-time requirements.

**Preprocessing Dependency**: Ridge performance critically depends on StandardScaler normalization. The fitted scaler from training (containing mean/variance parameters) must be preserved and applied identically in production to maintain prediction quality. This preprocessing consistency represents a common production ML pitfall addressed through shared artifact management.

### Shadow Mode Deployment Strategy

**Problem**: Deploy ML models safely in production without risking business impact from untested model behavior.

**Solution**: Shadow mode runs ALL 4 models on every transaction but only returns the selected model's decision (typically champion). All predictions get cached for post-hoc analysis, enabling:
- Risk-free model comparison with real production traffic
- Statistical validation of model performance before promotion
- A/B testing with deterministic traffic assignment (customer ID hashing)
- Business metric tracking (discrimination ratio, approval rates) alongside technical metrics

**Experiment Management**: Traffic allocation supports various experiment designs beyond 50/50 splits. Champion-challenger experiments might allocate 70% traffic to proven models, 30% to challengers. Multi-model comparisons can split traffic equally for comprehensive benchmarking. Gradual rollouts start at 5% challenger traffic, scaling as confidence builds.

This approach transforms production deployment from risky one-way decisions to reversible experiments that build institutional knowledge about what actually works in production contexts.

### Storage Abstraction for Evolution

Production ML systems evolve through distinct phases with different storage requirements: development prototyping, initial deployment, scaling growth, mature operations. Traditional tightly-coupled storage creates technical debt during phase transitions.

**Implementation**: Protocol-based interface defines operations (store prediction, retrieve experiments, analyze performance) independent of backend technology.

**Evolution Path**:
- **Development**: In-memory storage (Python deque) enables rapid iteration without infrastructure
- **Initial Production**: Redis caching (30-day TTL) + daily BigQuery batch balances cost and analytics
- **Future Scale**: Enhanced Redis clustering or BigQuery streaming when transaction volumes justify investment

This abstraction prevents premature optimization while ensuring seamless scaling when needed.

## Data Constraints and Synthetic Environment

The system operates entirely on Simtom-generated synthetic eCommerce data, creating unique challenges. Training dataset contains 1.9M transactions for 1,000 customers with only basic demographic and transaction features - no payment history beyond initial transaction. This eliminates traditional credit signals like payment velocity, account aging, or delinquency patterns.

**Feature Engineering Strategy**: Relies on transaction-time risk indicators:
- Customer demographics: credit score range, income bracket, age bracket
- Product characteristics: category, risk level, price
- Device trust signals: trusted device flags, device type
- Behavioral patterns: time on site, price comparison duration, purchase context

**Monitoring Implications**: Synthetic data distributions may drift differently than real customer behavior, requiring more aggressive monitoring thresholds and potentially faster retraining cadences than typical production fintech systems. The shadow mode architecture provides continuous validation of model performance against evolving synthetic distributions.

## Implementation Highlights

### Railway Deployment Architecture
- **Platform**: Railway PaaS with managed Redis co-located for sub-millisecond latency
- **Containerization**: Docker with Python 3.11-slim, all 4 models (~5MB) loaded at startup
- **Configuration**: Automatic environment detection via REDIS_URL presence
- **Hybrid Dependency Management**: Poetry for development, requirements.txt for deployment (Railway infrastructure constraints)

### MLflow Integration and Current Limitations
- **Tracking**: Every prediction generates MLflow run in background threads (non-blocking)
- **Current Backend**: SQLite stored in ephemeral container filesystem
- **Limitation**: All experiment data lost on container restart/redeployment
- **Phase 3 Solution**: Dedicated MLflow server with PostgreSQL backend for persistent storage, team access, and model registry capabilities

### Performance Characteristics
- **End-to-End Latency**: <20ms (target: <100ms)
- **Feature Engineering**: ~15ms (75% of latency)
- **4-Model Inference**: ~2ms total (10% of latency)
- **Redis Operations**: <1ms (Railway internal networking)

The latency breakdown reveals optimization opportunities: feature engineering dominates processing time through pandas operations and categorical encoding. Current implementation prioritizes maintainability over micro-optimization, accepting 15ms as reasonable given the 100ms budget. Future optimization could reduce this through Cython compilation, pre-computed lookups, or Rust-based extraction if business requirements tighten.

## Known Limitations and Future Work

### Current Limitations

**Feature Importance Anomalies**: Ridge coefficients cluster uniformly around 0.0009 across all 36 features, suggesting potential over-regularization or preprocessing artifacts. Key risk indicators (risk_score, credit_score_range) don't appear in top-weighted features as expected. This warrants investigation of StandardScaler normalization and Ridge alpha parameter tuning.

**Ensemble Probability Calibration**: VotingClassifier combines Ridge (uncalibrated decision_function) with Logistic/Elastic models (calibrated predict_proba), but averaging mixed outputs produces invalid probabilities. Current workaround averages only calibrated predictions. Proper solution wraps Ridge with sklearn's CalibratedClassifierCV during training.

**Ephemeral MLflow Storage**: Container restarts erase experiment history, preventing automated champion selection and long-term drift analysis.

**Data Quality**: Simtom synthetic data quality impacts model performance. Collaboration with Simtom team planned to improve default correlation patterns and feature distributions.

### Phase 3 Priorities

**Persistent MLflow Infrastructure**:
- Deploy MLflow server on Railway with PostgreSQL backend
- Enable team-wide access to experiment results
- Implement model registry for proper versioning (Development → Staging → Production)

**BigQuery Analytics Integration**:
- Long-term prediction storage and business intelligence
- Ground truth joining for model performance validation
- Data drift detection and retraining trigger automation

**Enhanced Feature Engineering**:
- Interaction features (risk_score × amount)
- Polynomial terms for non-linear relationships
- Target encoding for high-cardinality categoricals
- Systematic testing via shadow mode: deploy challenger with enhanced features, measure business metric improvement

## Related Documentation

**Architecture Patterns** (Platform-Wide):
- [Shadow Mode Controller Design](../../architecture/shadow_mode_controller.md) - Detailed A/B testing orchestration
- [ML Data Infrastructure](../../architecture/ml_data_infrastructure.md) - Complete data flow and storage strategy

**Model Details** (BNPL-Specific):
- [Known Issues](../../models/bnpl0925_known_issues.md) - Comprehensive limitation analysis
- [Production Artifacts](../../models/bnpl0925_production_artifacts.md) - Model export and deployment artifacts

**Implementation Reference**:
- `flit_ml/models/bnpl/predictor.py` - Multi-model inference (222 lines)
- `flit_ml/features/bnpl_feature_engineering.py` - Feature pipeline (810 lines)
- `flit_ml/core/shadow_controller.py` - Orchestration logic (797 lines)
- `flit_ml/api/bnpl_endpoints.py` - FastAPI integration (275 lines)
