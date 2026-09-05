# =============================================================================
# app.py
# Main Streamlit application
# =============================================================================

import streamlit as st

from ui import (
    inject_css,
    render_hero,
    render_input_form,
    render_placeholder,
    render_result,
)

from api_client import predict_via_api


# =============================================================================
# Page Configuration
# =============================================================================

st.set_page_config(
    page_title="Beverage Price Predictor",
    page_icon="🧃",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# =============================================================================
# Styling + Header
# =============================================================================

inject_css()
render_hero()


# =============================================================================
# Main Layout
# =============================================================================

left_col, right_col = st.columns(
    [3, 2],
    gap="large",
)


# =============================================================================
# Input Form
# =============================================================================

with left_col:
    inputs, predict_clicked = render_input_form()


# =============================================================================
# Prediction Panel
# =============================================================================

with right_col:

    if predict_clicked:

        try:

            with st.spinner(
                "Running prediction on AWS..."
            ):

                (
                    label,
                    confidence,
                    probabilities,
                    pred_class,
                ) = predict_via_api(inputs)

            render_result(
                label=label,
                confidence=confidence,
                probabilities=probabilities,
                pred_class=pred_class,
            )

        except Exception as error:

            st.error(
                "Prediction service is temporarily unavailable."
            )

            st.caption(
                f"Technical details: {error}"
            )

    else:

        render_placeholder()