# Model Card
## Hydraulic System Diagnostic Models

**Author:** Artur Melnyk

![Subsystem Model Architecture](../images/diagrams/04_model_architecture.png)

## Intended Use

The models are designed for:

- predictive maintenance
- hydraulic subsystem diagnostics
- industrial condition monitoring
- ML portfolio demonstration

The models are not intended for:

- autonomous safety-critical shutdown decisions
- direct industrial control systems
- production deployment without engineering validation

## Inputs

All models consume feature-engineered data from:

```text
data/processed/X_features_fe.parquet
```

The original feature-engineering stage generates approximately 130k+ features, but the final training process retains approximately 500 selected features.

Feature order is preserved using:

```text
artifacts/feature_index.json
```

## Outputs

| Model File | Output |
|---|---|
| `cooler_model.joblib` | Cooler condition |
| `valve_model.joblib` | Valve degradation |
| `pump_model.joblib` | Pump leakage |
| `accumulator_model.joblib` | Accumulator pressure |
| `stable_model.joblib` | Stable / unstable system state |

## Model Family

- algorithm: XGBoost
- learning type: supervised classification
- architecture: one independent model per subsystem

## Metrics

| Target | Accuracy | Macro F1 |
|---|---:|---:|
| Cooler | 0.995 | 0.995 |
| Valve | 0.659 | 0.574 |
| Pump | 0.956 | 0.948 |
| Accumulator | 0.943 | 0.934 |
| Stable | 0.943 | 0.936 |

Macro F1 is emphasized because several targets exhibit class imbalance.

## Known Limitation

The valve model underperforms relative to the other subsystem models.

```text
Valve degradation
    ↓
Weak sensor signature
    ↓
Low separability
    ↓
Lower predictive performance
```

This suggests a limitation in the underlying physical signal rather than a failure of the algorithm.

## Explainability

![Pump SHAP Beeswarm](../images/model_explain/pump_model_shap_beeswarm.png)

The project uses SHAP to provide:

- global feature importance
- subsystem-specific feature drivers
- comparison of feature importance across subsystem models

## Assumptions

The models assume:

- inference data follows the same distribution as the training data
- feature order matches `feature_index.json`
- the same preprocessing pipeline is used during prediction
- the deployed dashboard has been locally validated using saved test samples and feature-aligned inference inputs
