# Beverage Price Prediction — End-to-End AWS ML System

An end-to-end machine learning and MLOps project for predicting a consumer's preferred beverage price range from survey and behavioral attributes.

The project started as a local classification problem and was rebuilt on AWS to implement a production-oriented workflow covering data processing, model development, experiment tracking, managed training, model governance, serverless inference, API integration, monitoring, and automated retraining pipelines.
APP Link : https://beverage-price-prediction-aws.streamlit.app/
---

## Business Problem

Consumer beverage brands need to understand the price range customers are most likely to accept.

The objective is to classify respondents into one of four price segments:

- ₹50–100
- ₹100–150
- ₹150–200
- ₹200–250

The prediction uses demographic, consumption, brand, packaging, purchase-channel, and behavioral attributes.

Potential business applications include:

- pricing strategy
- customer segmentation
- targeted promotions
- product positioning
- market research

---

## Dataset

The raw survey dataset contains approximately 30,000 consumer responses.

After deterministic data-quality processing:

| Item | Value |
|---|---:|
| Raw rows | 30,010 |
| Duplicate rows removed | 10 |
| Logical outliers removed | 44 |
| Final cleaned rows | 29,956 |
| Final model features | 27 |
| Target classes | 4 |

The preprocessing workflow includes:

- duplicate removal
- category and spelling standardization
- logical outlier filtering
- missing-value handling
- categorical encoding
- engineered behavioral features
- consistent training/inference feature schema

Missing-value imputation and encoding are fitted only on training data to avoid leakage.

---

## Model Development

Multiple classification algorithms were evaluated using a leakage-safe model-development process.

Candidate models included:

- Logistic Regression
- Random Forest
- LightGBM
- XGBoost

Hyperparameter optimization was performed using Optuna.

XGBoost was selected as the production model based primarily on Macro F1 because the objective required balanced performance across all four price classes.

### Final Holdout Performance

The final evaluation was performed on an untouched stratified holdout set of 7,489 observations.

| Metric | Score |
|---|---:|
| Accuracy | 0.9246 |
| Macro Precision | 0.9237 |
| Macro Recall | 0.9235 |
| Macro F1 | **0.9236** |
| Weighted F1 | 0.9247 |
| Ordinal MAE | **0.0754** |

The low ordinal MAE also indicates that most classification errors occur between neighboring price ranges rather than distant price bands.

---

## AWS Architecture

```mermaid
flowchart TD

    A[Raw Survey Data] --> B[Amazon S3]

    B --> C[SageMaker Processing]
    C --> D[Cleaned Dataset]

    D --> E[Train / Test Split]
    E --> F[Candidate XGBoost Training]
    E --> G[Untouched Holdout Test]

    F --> H[SageMaker Model Evaluation]
    G --> H

    H --> I{Macro F1 >= 0.90?}

    I -->|Yes| J[Full-Data Production Refit]
    I -->|No| K[Stop Pipeline]

    J --> L[SageMaker Model Registry]
    L --> M[Manual Model Approval]

    M --> N[SageMaker Serverless Endpoint]
    N --> O[AWS Lambda]
    O --> P[Amazon API Gateway]
    P --> Q[Streamlit Application]

    R[MLflow Experiment Tracking] -.-> F
    S[CloudWatch Monitoring] -.-> O
    S -.-> N
