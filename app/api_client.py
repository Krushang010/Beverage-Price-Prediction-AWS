# =============================================================================
# api_client.py
# Secure client for AWS API Gateway prediction endpoint
# =============================================================================

import os

import requests
import streamlit as st

from config import (
    PREDICT_API_URL,
    PRICE_LABELS,
)


# =============================================================================
# API Secret
# =============================================================================

def get_api_secret() -> str:
    """
    Get the shared API secret.

    Priority:
    1. Environment variable
    2. Streamlit Community Cloud secrets
    """

    # Local / server environment variable
    secret = os.getenv("API_SHARED_SECRET")

    if secret:
        return secret

    # Streamlit Community Cloud
    try:
        secret = st.secrets["API_SHARED_SECRET"]

        if secret:
            return str(secret)

    except Exception:
        pass

    raise RuntimeError(
        "API_SHARED_SECRET is not configured."
    )


# =============================================================================
# Age Group
# =============================================================================

def get_age_group(age: int) -> str:
    """Convert age into the model's expected age group."""

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


# =============================================================================
# Build Request Payload
# =============================================================================

def build_payload(inputs: dict) -> dict:
    """
    Convert Streamlit inputs into the raw JSON schema
    expected by the SageMaker inference pipeline.
    """

    record = {

        "respondent_id": "STREAMLIT_USER",

        "age": inputs["age"],

        "gender": inputs["gender"],

        "zone": inputs["zone"],

        "occupation": inputs["occupation"],

        "income_levels": inputs["income"],

        "consume_frequency(weekly)":
            inputs["consume_frequency"],

        "current_brand":
            inputs["current_brand"],

        "preferable_consumption_size":
            inputs["size"],

        "awareness_of_other_brands":
            inputs["awareness"],

        "reasons_for_choosing_brands":
            inputs["reasons"],

        "flavor_preference":
            inputs["flavor"],

        "purchase_channel":
            inputs["channel"],

        "packaging_preference":
            inputs["packaging"],

        "health_concerns":
            inputs["health"],

        "typical_consumption_situations":
            inputs["situation"],

        "age_group":
            get_age_group(inputs["age"]),
    }

    return {
        "instances": [record]
    }


# =============================================================================
# Prediction API Call
# =============================================================================

def predict_via_api(inputs: dict):
    """
    Send prediction request to API Gateway.

    Flow:
    Streamlit
        ↓
    API Gateway
        ↓
    Lambda authentication
        ↓
    SageMaker Serverless Endpoint
        ↓
    XGBoost prediction
    """

    payload = build_payload(inputs)

    api_secret = get_api_secret()

    response = requests.post(

        PREDICT_API_URL,

        json=payload,

        headers={
            "X-API-Key": api_secret,
            "Content-Type": "application/json",
        },

        timeout=35,
    )


    # Raise error for 401 / 403 / 500 etc.
    response.raise_for_status()


    # -------------------------------------------------------------------------
    # Parse response
    # -------------------------------------------------------------------------

    data = response.json()

    prediction = data["predictions"][0]


    pred_class = int(
        prediction["predicted_class"]
    )


    confidence = (
        float(prediction["confidence"])
        * 100
    )


    probability_map = (
        prediction["probabilities"]
    )


    probabilities = [

        float(
            probability_map["50-100"]
        ),

        float(
            probability_map["100-150"]
        ),

        float(
            probability_map["150-200"]
        ),

        float(
            probability_map["200-250"]
        ),
    ]


    label = PRICE_LABELS[
        pred_class
    ]


    return (
        label,
        confidence,
        probabilities,
        pred_class,
    )