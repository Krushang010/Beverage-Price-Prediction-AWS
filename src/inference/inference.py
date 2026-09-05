
import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


# --------------------------------------------------
# Allow access to sibling preprocessing package
# --------------------------------------------------
CODE_DIR = Path(__file__).resolve().parent

if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from preprocessing.feature_utils import transform_features


# --------------------------------------------------
# Load model
# --------------------------------------------------

def model_fn(model_dir):

    bundle_path = os.path.join(
        model_dir,
        "model_bundle.joblib"
    )

    bundle = joblib.load(bundle_path)

    return bundle


# --------------------------------------------------
# Deserialize request
# --------------------------------------------------

def input_fn(request_body, request_content_type):

    if request_content_type != "application/json":
        raise ValueError(
            f"Unsupported content type: "
            f"{request_content_type}"
        )

    if isinstance(
        request_body,
        (bytes, bytearray)
    ):
        request_body = request_body.decode("utf-8")

    payload = json.loads(request_body)

    # Supported formats:
    #
    # {"instances": [{...}, {...}]}
    #
    # OR
    #
    # {...single record...}
    #
    # OR
    #
    # [{...}, {...}]

    if isinstance(payload, dict):

        if "instances" in payload:
            records = payload["instances"]
        else:
            records = [payload]

    elif isinstance(payload, list):
        records = payload

    else:
        raise ValueError(
            "JSON payload must be a record, "
            "list of records, or contain "
            "'instances'."
        )

    if not records:
        raise ValueError(
            "No prediction records supplied."
        )

    return pd.DataFrame(records)


# --------------------------------------------------
# Predict
# --------------------------------------------------

def predict_fn(input_data, bundle):

    preprocessing = bundle["preprocessing"]
    model = bundle["model"]

    inverse_price_map = (
        bundle["inverse_price_map"]
    )

    # Apply EXACT preprocessing learned
    # during training.
    X = transform_features(
        input_data,
        preprocessing
    )

    predictions = model.predict(X)

    probabilities = model.predict_proba(X)

    model_classes = model.classes_

    results = []

    for prediction, probability_row in zip(
        predictions,
        probabilities
    ):

        predicted_class = int(prediction)

        probability_map = {
            inverse_price_map[int(class_id)]:
                float(probability)

            for class_id, probability
            in zip(
                model_classes,
                probability_row
            )
        }

        results.append({
            "predicted_class":
                predicted_class,

            "predicted_price_range":
                inverse_price_map[
                    predicted_class
                ],

            "confidence":
                float(
                    np.max(
                        probability_row
                    )
                ),

            "probabilities":
                probability_map,
        })

    return results


# --------------------------------------------------
# Serialize response
# --------------------------------------------------

def output_fn(
    prediction,
    accept
):

    if accept not in (
        "application/json",
        "*/*"
    ):
        raise ValueError(
            f"Unsupported accept type: {accept}"
        )

    response = {
        "predictions": prediction
    }

    return (
        json.dumps(response),
        "application/json"
    )
