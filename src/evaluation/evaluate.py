
import os
import json
import tarfile
from pathlib import Path

import joblib
import pandas as pd
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from preprocessing.feature_utils import transform_features


MODEL_DIR = "/opt/ml/processing/model"
TEST_DIR = "/opt/ml/processing/test"
OUTPUT_DIR = "/opt/ml/processing/evaluation"

TARGET = "price_range"


def find_file(root, pattern):
    files = list(Path(root).rglob(pattern))

    if not files:
        raise FileNotFoundError(
            f"Could not find {pattern} under {root}"
        )

    return files[0]


def main():

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    # --------------------------------------------------
    # Load trained model artifact
    # --------------------------------------------------

    model_tar = find_file(
        MODEL_DIR,
        "model.tar.gz",
    )

    extracted_dir = "/tmp/beverage-model"

    os.makedirs(
        extracted_dir,
        exist_ok=True,
    )

    with tarfile.open(
        model_tar,
        "r:gz",
    ) as tar:
        tar.extractall(extracted_dir)


    bundle_path = find_file(
        extracted_dir,
        "model_bundle.joblib",
    )

    bundle = joblib.load(
        bundle_path
    )

    model = bundle["model"]
    preprocessing = bundle["preprocessing"]
    price_map = bundle["price_map"]


    # --------------------------------------------------
    # Load untouched test data
    # --------------------------------------------------

    test_file = find_file(
        TEST_DIR,
        "*.csv",
    )

    test_df = pd.read_csv(
        test_file
    )

    if TARGET not in test_df.columns:
        raise ValueError(
            f"Target column {TARGET} missing."
        )


    # --------------------------------------------------
    # Prepare target
    # --------------------------------------------------

    y_true = (
        test_df[TARGET]
        .map(price_map)
        .astype(int)
    )


    # --------------------------------------------------
    # Prepare model features
    # --------------------------------------------------

    X_raw = test_df.drop(
        columns=[
            TARGET,
            "respondent_id",
        ],
        errors="ignore",
    )

    X_test = transform_features(
        X_raw,
        preprocessing,
    )


    # --------------------------------------------------
    # Predict
    # --------------------------------------------------

    y_pred = model.predict(
        X_test
    ).astype(int)


    # --------------------------------------------------
    # Metrics
    # --------------------------------------------------

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    macro_precision = precision_score(
        y_true,
        y_pred,
        average="macro",
    )

    macro_recall = recall_score(
        y_true,
        y_pred,
        average="macro",
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
    )

    weighted_f1 = f1_score(
        y_true,
        y_pred,
        average="weighted",
    )

    ordinal_mae = np.abs(
        y_true.to_numpy() - y_pred
    ).mean()


    report = {
        "metrics": {
            "accuracy": float(accuracy),
            "macro_precision": float(macro_precision),
            "macro_recall": float(macro_recall),
            "macro_f1": float(macro_f1),
            "weighted_f1": float(weighted_f1),
            "ordinal_mae": float(ordinal_mae),
        },
        "test_rows": int(len(test_df)),
        "feature_count": int(X_test.shape[1]),
    }


    # --------------------------------------------------
    # Save evaluation report
    # --------------------------------------------------

    evaluation_path = os.path.join(
        OUTPUT_DIR,
        "evaluation.json",
    )

    with open(
        evaluation_path,
        "w",
    ) as f:
        json.dump(
            report,
            f,
            indent=4,
        )


    # --------------------------------------------------
    # Save confusion matrix
    # --------------------------------------------------

    cm = confusion_matrix(
        y_true,
        y_pred,
    )

    cm_df = pd.DataFrame(
        cm
    )

    cm_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "confusion_matrix.csv",
        ),
        index=False,
    )


    print(
        json.dumps(
            report,
            indent=4,
        )
    )

    print(
        "Evaluation complete ✅"
    )


if __name__ == "__main__":
    main()
