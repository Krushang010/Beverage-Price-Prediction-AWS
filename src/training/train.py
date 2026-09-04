
import glob
import json
import os
import sys

from pathlib import Path
from datetime import datetime, timezone

import joblib
import pandas as pd
import sklearn
import xgboost

from xgboost import XGBClassifier


# --------------------------------------------------
# Make src/ importable inside SageMaker container
# --------------------------------------------------

SRC_DIR = Path(__file__).resolve().parents[1]

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from preprocessing.feature_utils import (
    PRICE_MAP,
    fit_transform_features,
)


def find_file(directory, extension):
    matches = glob.glob(
        os.path.join(directory, f"*{extension}")
    )

    if not matches:
        raise FileNotFoundError(
            f"No {extension} file found in {directory}"
        )

    return matches[0]


def main():

    training_dir = os.environ["SM_CHANNEL_TRAINING"]
    config_dir = os.environ["SM_CHANNEL_CONFIG"]
    model_dir = os.environ["SM_MODEL_DIR"]

    # --------------------------------------------------
    # Load managed S3 inputs
    # --------------------------------------------------

    data_path = find_file(
        training_dir,
        ".csv"
    )

    params_path = find_file(
        config_dir,
        ".json"
    )

    df = pd.read_csv(data_path)

    with open(params_path, "r") as f:
        best_params = json.load(f)

    print(f"Training rows: {len(df)}")

    # --------------------------------------------------
    # Target
    # --------------------------------------------------

    y = df["price_range"].map(PRICE_MAP)

    if y.isna().any():
        raise ValueError(
            "Unexpected target category detected."
        )

    # --------------------------------------------------
    # Fit production preprocessing on ALL labeled data
    # --------------------------------------------------

    X, preprocessing_artifacts = (
        fit_transform_features(df)
    )

    print(f"Final feature count: {X.shape[1]}")

    if X.isna().sum().sum() != 0:
        raise ValueError(
            "Missing values remain after preprocessing."
        )

    # --------------------------------------------------
    # Frozen Optuna-selected XGBoost configuration
    # --------------------------------------------------

    model = XGBClassifier(
        **best_params,
        random_state=42,
        n_jobs=-1,
        eval_metric="mlogloss",
        verbosity=0,
    )

    model.fit(X, y)

    # --------------------------------------------------
    # Production artifact bundle
    # --------------------------------------------------

    bundle = {
        "model": model,
        "preprocessing": preprocessing_artifacts,
        "price_map": PRICE_MAP,
        "inverse_price_map": {
            v: k for k, v in PRICE_MAP.items()
        },
    }

    os.makedirs(
        model_dir,
        exist_ok=True
    )

    bundle_path = os.path.join(
        model_dir,
        "model_bundle.joblib"
    )

    joblib.dump(
        bundle,
        bundle_path
    )

    # Native XGBoost artifact as an additional portable copy
    model.save_model(
        os.path.join(
            model_dir,
            "xgboost_model.json"
        )
    )

    metadata = {
        "model": "XGBoost",
        "purpose": "production_refit",
        "training_rows": int(len(df)),
        "feature_count": int(X.shape[1]),
        "hyperparameters": best_params,
        "xgboost_version": xgboost.__version__,
        "sklearn_version": sklearn.__version__,
        "trained_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "official_holdout_metrics": {
            "accuracy": 0.9246,
            "macro_f1": 0.9236,
            "ordinal_mae": 0.0754,
        },
    }

    with open(
        os.path.join(
            model_dir,
            "training_metadata.json"
        ),
        "w"
    ) as f:
        json.dump(
            metadata,
            f,
            indent=4
        )

    print("Production model training complete.")
    print(f"Rows used: {len(df)}")
    print(f"Features: {X.shape[1]}")
    print(f"Artifact: {bundle_path}")


if __name__ == "__main__":
    main()
