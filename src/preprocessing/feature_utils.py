
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder


PRICE_MAP = {
    "50-100": 0,
    "100-150": 1,
    "150-200": 2,
    "200-250": 3,
}

AGE_GROUP_MAP = {
    "18-25": 1,
    "26-35": 2,
    "36-45": 3,
    "46-55": 4,
    "56-70": 5,
}

INCOME_MAP = {
    "Not Reported": 0,
    "<10L": 1,
    "10L - 15L": 2,
    "16L - 25L": 3,
    "26L - 35L": 4,
    "> 35L": 5,
}

HEALTH_MAP = {
    "Low (Not very concerned)": 1,
    "Medium (Moderately health-conscious)": 2,
    "High (Very health-conscious)": 3,
}

FREQ_MAP = {
    "0-2 times": 1,
    "3-4 times": 2,
    "5-7 times": 3,
}

SIZE_MAP = {
    "Small (250 ml)": 1,
    "Medium (500 ml)": 2,
    "Large (1 L)": 3,
}

CF_MAP = FREQ_MAP

AB_MAP = {
    "0 to 1": 1,
    "2 to 4": 2,
    "above 4": 3,
}

ZONE_MAP = {
    "Rural": 1,
    "Semi-Urban": 2,
    "Urban": 3,
    "Metro": 4,
}

IMPUTE_COLS = [
    "consume_frequency(weekly)",
    "purchase_channel",
]


def _business_features(df):
    df = df.copy()

    # Idempotent deterministic cleaning
    df["zone"] = df["zone"].replace({
        "urbna": "Urban",
        "Metor": "Metro",
    })

    df["current_brand"] = df["current_brand"].replace({
        "newcomer": "Newcomer",
        "Establishd": "Established",
    })

    df["income_levels"] = (
        df["income_levels"]
        .fillna("Not Reported")
        .str.strip()
    )

    # Recompute from raw age so serving uses same rule
    df["age_group"] = pd.cut(
        df["age"],
        bins=[17, 25, 35, 45, 55, 70],
        labels=[
            "18-25",
            "26-35",
            "36-45",
            "46-55",
            "56-70",
        ],
        include_lowest=True
    ).astype(str)

    df["zone_num"] = df["zone"].map(ZONE_MAP)
    df["income_num"] = df["income_levels"].map(INCOME_MAP)

    df["zas_score"] = (
        df["zone_num"] * df["income_num"]
    )

    df["bsi"] = (
        (df["current_brand"] != "Established")
        &
        (
            df["reasons_for_choosing_brands"]
            .isin(["Price", "Quality"])
        )
    ).astype(int)

    df["cf_num"] = (
        df["consume_frequency(weekly)"].map(CF_MAP)
    )

    df["ab_num"] = (
        df["awareness_of_other_brands"].map(AB_MAP)
    )

    df["cf_ab_score"] = (
        df["cf_num"]
        /
        (df["cf_num"] + df["ab_num"])
    ).round(2)

    # Fixed ordinal mappings
    df["age_group"] = df["age_group"].map(AGE_GROUP_MAP)
    df["income_levels"] = df["income_levels"].map(INCOME_MAP)
    df["health_concerns"] = df["health_concerns"].map(HEALTH_MAP)

    df["consume_frequency(weekly)"] = (
        df["consume_frequency(weekly)"].map(FREQ_MAP)
    )

    df["preferable_consumption_size"] = (
        df["preferable_consumption_size"].map(SIZE_MAP)
    )

    drop_cols = [
        "respondent_id",
        "price_range",
        "age",
        "cf_num",
        "ab_num",
        "zone_num",
        "income_num",
    ]

    df = df.drop(
        columns=[c for c in drop_cols if c in df.columns]
    )

    return df


def fit_transform_features(df):
    data = df.copy()

    imputer = SimpleImputer(
        strategy="most_frequent"
    )

    data[IMPUTE_COLS] = imputer.fit_transform(
        data[IMPUTE_COLS]
    )

    data = _business_features(data)

    cat_cols = data.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    num_cols = [
        c for c in data.columns
        if c not in cat_cols
    ]

    ohe = OneHotEncoder(
        handle_unknown="ignore",
        drop="first",
        sparse_output=False,
    )

    encoded = ohe.fit_transform(
        data[cat_cols]
    )

    encoded_cols = ohe.get_feature_names_out(
        cat_cols
    )

    encoded_df = pd.DataFrame(
        encoded,
        columns=encoded_cols,
        index=data.index,
    )

    X = pd.concat(
        [data[num_cols], encoded_df],
        axis=1,
    )

    artifacts = {
        "imputer": imputer,
        "ohe": ohe,
        "cat_cols": cat_cols,
        "num_cols": num_cols,
        "feature_names": X.columns.tolist(),
    }

    return X, artifacts


def transform_features(df, artifacts):
    data = df.copy()

    data[IMPUTE_COLS] = (
        artifacts["imputer"].transform(
            data[IMPUTE_COLS]
        )
    )

    data = _business_features(data)

    encoded = artifacts["ohe"].transform(
        data[artifacts["cat_cols"]]
    )

    encoded_df = pd.DataFrame(
        encoded,
        columns=artifacts["ohe"].get_feature_names_out(
            artifacts["cat_cols"]
        ),
        index=data.index,
    )

    X = pd.concat(
        [
            data[artifacts["num_cols"]],
            encoded_df
        ],
        axis=1,
    )

    # Guarantee training schema/order
    X = X.reindex(
        columns=artifacts["feature_names"],
        fill_value=0,
    )

    return X
