# =============================================================================
# config.py
# Frontend constants for Beverage Price Range Prediction
# =============================================================================

PREDICT_API_URL = (
    "https://yogsvlf97e.execute-api.ap-south-1.amazonaws.com/predict"
)

PRICE_LABELS = {
    0: "₹50–100",
    1: "₹100–150",
    2: "₹150–200",
    3: "₹200–250",
}

GENDER_OPTIONS = ["M", "F"]

ZONE_OPTIONS = [
    "Metro",
    "Urban",
    "Semi-Urban",
    "Rural",
]

OCCUPATION_OPTIONS = [
    "Working Professional",
    "Student",
    "Entrepreneur",
    "Retired",
]

INCOME_OPTIONS = [
    "<10L",
    "10L - 15L",
    "16L - 25L",
    "26L - 35L",
    "> 35L",
    "Not Reported",
]

FREQ_OPTIONS = [
    "0-2 times",
    "3-4 times",
    "5-7 times",
]

BRAND_OPTIONS = [
    "Newcomer",
    "Established",
]

SIZE_OPTIONS = [
    "Small (250 ml)",
    "Medium (500 ml)",
    "Large (1 L)",
]

AWARENESS_OPTIONS = [
    "0 to 1",
    "2 to 4",
    "above 4",
]

REASONS_OPTIONS = [
    "Price",
    "Quality",
    "Availability",
    "Brand Reputation",
]

FLAVOR_OPTIONS = [
    "Traditional",
    "Exotic",
]

CHANNEL_OPTIONS = [
    "Online",
    "Retail Store",
]

PACKAGING_OPTIONS = [
    "Simple",
    "Premium",
    "Eco-Friendly",
]

HEALTH_OPTIONS = [
    "Low (Not very concerned)",
    "Medium (Moderately health-conscious)",
    "High (Very health-conscious)",
]

SITUATION_OPTIONS = [
    "Active (eg. Sports, gym)",
    "Social (eg. Parties)",
    "Casual (eg. At home)",
]