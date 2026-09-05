# =============================================================================
# api_client.py
# Sends Streamlit inputs to the AWS API Gateway prediction endpoint
# =============================================================================

import requests

from config import PREDICT_API_URL, PRICE_LABELS


def get_age_group(age: int) -> str:
    """Convert raw age into the age group used by the model."""

    if age <= 25:
        return "18-25"
    elif age <= 35:
        return "26-35"
    elif age <= 45:
        return "36-45"
    elif age <= 55:
        return "46-55"
    else:
        return "56-70"


def build_payload(inputs: dict) -> dict:
    """
    Convert Streamlit form inputs into the JSON payload
    expected by the SageMaker inference endpoint.
    """

    record = {
        "respondent_id": "STREAMLIT_USER",
        "age": inputs["age"],
        "gender": inputs["gender"],
        "zone": inputs["zone"],
        "occupation": inputs["occupation"],
        "income_levels": inputs["income"],
        "consume_frequency(weekly)": inputs["consume_frequency"],
        "current_brand": inputs["current_brand"],
        "preferable_consumption_size": inputs["size"],
        "awareness_of_other_brands": inputs["awareness"],
        "reasons_for_choosing_brands": inputs["reasons"],
        "flavor_preference": inputs["flavor"],
        "purchase_channel": inputs["channel"],
        "packaging_preference": inputs["packaging"],
        "health_concerns": inputs["health"],
        "typical_consumption_situations": inputs["situation"],
        "age_group": get_age_group(inputs["age"]),
    }

    return {
        "instances": [record]
    }


def predict_via_api(inputs: dict):
    """
    Send the prediction request to API Gateway.

    Returns
    -------
    label : str
    confidence : float
    probabilities : list
    pred_class : int
    """

    payload = build_payload(inputs)

    response = requests.post(
        PREDICT_API_URL,
        json=payload,
        timeout=35,
    )

    response.raise_for_status()

    data = response.json()

    prediction = data["predictions"][0]

    pred_class = int(
        prediction["predicted_class"]
    )

    confidence = (
        float(prediction["confidence"]) * 100
    )

    probability_map = prediction["probabilities"]

    probabilities = [
        float(probability_map["50-100"]),
        float(probability_map["100-150"]),
        float(probability_map["150-200"]),
        float(probability_map["200-250"]),
    ]

    label = PRICE_LABELS[pred_class]

    return (
        label,
        confidence,
        probabilities,
        pred_class,
    )