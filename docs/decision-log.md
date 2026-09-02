# AWS Beverage ML — Engineering Decision Log

## AWS Region
- Region: ap-south-1 (Mumbai)
- Reason: Primary project region selected for this build.

## IAM
- Root account used only for account-level setup.
- Daily AWS work uses IAM admin user.
- SageMaker uses a separate execution role.
- Custom S3 access granted explicitly to the SageMaker execution role.
- Learning: IAM user permissions and SageMaker execution-role permissions are separate.

## Amazon S3
- Bucket: krushang-beverage-ml-2026
- Block Public Access: Enabled.
- ACLs: Disabled / Bucket owner enforced.
- Default encryption: SSE-S3.
- Versioning: Enabled.
- Object Lock: Disabled.
- Prefixes:
  - raw/
  - processed/
  - models/
  - evaluation/

## Source Control
- GitHub is the source of truth for project code.
- Data and model binaries are not committed to Git.
- .ipynb_checkpoints, credentials, CSV/XLSX data and model binaries are ignored.

## Python Environment
- Development environment: SageMaker Distribution in JupyterLab.
- No project .venv inside SageMaker Studio.
- Direct project dependencies are pinned in requirements.txt.
- Managed Processing and Training jobs will use isolated runtime environments.

## Modeling
Candidate models:
1. Logistic Regression
2. Random Forest
3. XGBoost
4. LightGBM

Model selection will be based on measured validation performance, not the previously selected local model.

## Data Processing
- Raw data remains immutable in raw/.
- SageMaker Processing will create reproducible processed outputs.
- Learned transformations must be fitted on training data only.
- Final test data must not be used for hyperparameter tuning or model selection.

## MLOps
Planned:
- Git/GitHub for code versioning.
- S3 for datasets and artifacts.
- MLflow for experiment tracking.
- SageMaker Model Registry for model versions.
- CloudWatch for operational logging/monitoring.
- SageMaker Pipelines after the manual workflow is validated.
