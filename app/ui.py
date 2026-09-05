# =============================================================================
# ui.py
# Streamlit UI components for Beverage Price Range Prediction
# =============================================================================

import streamlit as st

from config import (
    PRICE_LABELS,
    GENDER_OPTIONS,
    ZONE_OPTIONS,
    OCCUPATION_OPTIONS,
    INCOME_OPTIONS,
    FREQ_OPTIONS,
    BRAND_OPTIONS,
    SIZE_OPTIONS,
    AWARENESS_OPTIONS,
    REASONS_OPTIONS,
    FLAVOR_OPTIONS,
    CHANNEL_OPTIONS,
    PACKAGING_OPTIONS,
    HEALTH_OPTIONS,
    SITUATION_OPTIONS,
)


# =============================================================================
# Styling
# =============================================================================

def inject_css():

    st.html("""
    <style>

    .block-container {
        max-width: 1400px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .hero-box {
        background: linear-gradient(
            135deg,
            #101828 0%,
            #1849a9 55%,
            #1570ef 100%
        );
        padding: 2rem 2.2rem;
        border-radius: 18px;
        color: white;
        margin-bottom: 1.8rem;
    }

    .hero-badge {
        display: inline-block;
        padding: 5px 12px;
        border-radius: 20px;
        border: 1px solid rgba(255,255,255,0.25);
        background: rgba(255,255,255,0.10);
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        margin-bottom: 0.8rem;
    }

    .hero-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.4rem;
    }

    .hero-subtitle {
        font-size: 0.95rem;
        opacity: 0.82;
    }

    .section-title {
        margin-top: 1rem;
        margin-bottom: 0.75rem;
        padding-bottom: 0.45rem;
        border-bottom: 1px solid #eaecf0;
        color: #98a2b3;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }

    .prediction-card {
        background: linear-gradient(
            135deg,
            #1849a9 0%,
            #1570ef 100%
        );
        border-radius: 18px;
        padding: 2rem 1.5rem;
        color: white;
        text-align: center;
        margin-bottom: 1.3rem;
    }

    .prediction-label {
        font-size: 0.75rem;
        opacity: 0.8;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    .prediction-price {
        font-size: 3rem;
        font-weight: 700;
        margin: 0.35rem 0;
    }

    .prediction-confidence {
        font-size: 0.9rem;
        opacity: 0.88;
    }

    .placeholder-box {
        border: 1.5px dashed #475467;
        border-radius: 18px;
        padding: 4rem 1rem;
        text-align: center;
    }

    .placeholder-icon {
        font-size: 2.7rem;
    }

    .placeholder-text {
        color: #98a2b3;
        font-size: 0.92rem;
        margin-top: 0.8rem;
    }

    .architecture-box {
        background: #f2f4f7;
        border-radius: 10px;
        padding: 0.8rem;
        margin-top: 1.3rem;
        text-align: center;
        color: #475467;
        font-size: 0.76rem;
    }

    </style>
    """)


# =============================================================================
# Hero
# =============================================================================

def render_hero():

    st.html("""
    <div class="hero-box">
        <div class="hero-badge">AWS • MACHINE LEARNING</div>

        <div class="hero-title">
            🧃 Beverage Price Range Predictor
        </div>

        <div class="hero-subtitle">
            Predict a consumer's preferred beverage price range
            using an XGBoost model deployed on AWS SageMaker.
        </div>
    </div>
    """)


# =============================================================================
# Input Form
# =============================================================================

def render_input_form():

    with st.form("beverage_prediction_form"):

        st.html(
            '<div class="section-title">👤 Demographics</div>'
        )

        col1, col2 = st.columns(2)

        with col1:
            age = st.number_input(
                "Age",
                min_value=18,
                max_value=70,
                value=30,
                step=1,
            )

            zone = st.selectbox(
                "Zone",
                ZONE_OPTIONS,
            )

        with col2:
            gender = st.selectbox(
                "Gender",
                GENDER_OPTIONS,
            )

            occupation = st.selectbox(
                "Occupation",
                OCCUPATION_OPTIONS,
            )


        st.html(
            '<div class="section-title">💰 Financial Profile</div>'
        )

        col1, col2 = st.columns(2)

        with col1:
            income = st.selectbox(
                "Income Level",
                INCOME_OPTIONS,
            )

        with col2:
            consume_frequency = st.selectbox(
                "Weekly Consumption Frequency",
                FREQ_OPTIONS,
            )


        st.html(
            '<div class="section-title">🏷️ Brand & Product Preferences</div>'
        )

        col1, col2 = st.columns(2)

        with col1:
            current_brand = st.selectbox(
                "Current Brand",
                BRAND_OPTIONS,
            )

            awareness = st.selectbox(
                "Awareness of Other Brands",
                AWARENESS_OPTIONS,
            )

            flavor = st.selectbox(
                "Flavor Preference",
                FLAVOR_OPTIONS,
            )

        with col2:
            size = st.selectbox(
                "Preferred Consumption Size",
                SIZE_OPTIONS,
            )

            reasons = st.selectbox(
                "Primary Reason for Choosing Brand",
                REASONS_OPTIONS,
            )

            channel = st.selectbox(
                "Purchase Channel",
                CHANNEL_OPTIONS,
            )


        st.html(
            '<div class="section-title">🌿 Lifestyle & Health</div>'
        )

        col1, col2 = st.columns(2)

        with col1:
            packaging = st.selectbox(
                "Packaging Preference",
                PACKAGING_OPTIONS,
            )

            health = st.selectbox(
                "Health Concerns",
                HEALTH_OPTIONS,
            )

        with col2:
            situation = st.selectbox(
                "Typical Consumption Situation",
                SITUATION_OPTIONS,
            )

        st.write("")

        predict_clicked = st.form_submit_button(
            "🔮 Predict Price Range",
            type="primary",
            use_container_width=True,
        )


    inputs = {
        "age": int(age),
        "gender": gender,
        "zone": zone,
        "occupation": occupation,
        "income": income,
        "consume_frequency": consume_frequency,
        "current_brand": current_brand,
        "size": size,
        "awareness": awareness,
        "reasons": reasons,
        "flavor": flavor,
        "channel": channel,
        "packaging": packaging,
        "health": health,
        "situation": situation,
    }

    return inputs, predict_clicked


# =============================================================================
# Placeholder
# =============================================================================

def render_placeholder():

    st.html(
        '<div class="section-title">📊 Prediction Result</div>'
    )

    st.html("""
    <div class="placeholder-box">
        <div class="placeholder-icon">🔮</div>

        <div class="placeholder-text">
            Complete the consumer profile and click
            <strong>Predict Price Range</strong>
            to generate a prediction.
        </div>
    </div>
    """)


# =============================================================================
# Result
# =============================================================================

def render_result(
    label: str,
    confidence: float,
    probabilities: list,
    pred_class: int,
):

    st.html(
        '<div class="section-title">📊 Prediction Result</div>'
    )

    st.html(
        f"""
        <div class="prediction-card">
            <div class="prediction-label">
                Predicted Price Range
            </div>

            <div class="prediction-price">
                {label}
            </div>

            <div class="prediction-confidence">
                Model Confidence: {confidence:.1f}%
            </div>
        </div>
        """
    )

    st.html(
        '<div class="section-title">Class Probabilities</div>'
    )

    for class_index, probability in enumerate(probabilities):

        probability = float(probability)

        price_label = PRICE_LABELS[class_index]

        col1, col2 = st.columns(
            [1.2, 4]
        )

        with col1:

            if class_index == pred_class:
                st.markdown(
                    f"**{price_label}**"
                )
            else:
                st.write(price_label)

        with col2:

            st.progress(
                min(
                    max(probability, 0.0),
                    1.0,
                )
            )

            st.caption(
                f"{probability * 100:.1f}%"
            )

    st.html("""
    <div class="architecture-box">
        Streamlit → API Gateway → AWS Lambda →
        SageMaker Serverless Endpoint → XGBoost
    </div>
    """)