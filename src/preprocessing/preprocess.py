
import argparse
import json
import os

import pandas as pd


EXPECTED_COLUMNS = [
    "respondent_id",
    "age",
    "gender",
    "zone",
    "occupation",
    "income_levels",
    "consume_frequency(weekly)",
    "current_brand",
    "preferable_consumption_size",
    "awareness_of_other_brands",
    "reasons_for_choosing_brands",
    "flavor_preference",
    "purchase_channel",
    "packaging_preference",
    "health_concerns",
    "typical_consumption_situations",
    "price_range",
]


def clean_data(df):

    df = df.copy()

    rows_raw = len(df)

    # -----------------------------
    # Schema validation
    # -----------------------------
    missing_columns = set(EXPECTED_COLUMNS) - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    # -----------------------------
    # Exact duplicates
    # -----------------------------
    duplicate_rows = int(df.duplicated().sum())

    df = df.drop_duplicates().copy()

    # -----------------------------
    # Confirmed categorical errors
    # -----------------------------
    df["zone"] = df["zone"].replace({
        "urbna": "Urban",
        "Metor": "Metro"
    })

    df["current_brand"] = df["current_brand"].replace({
        "newcomer": "Newcomer",
        "Establishd": "Established"
    })

    # -----------------------------
    # Business-semantic missingness
    # -----------------------------
    df["income_levels"] = (
        df["income_levels"]
        .fillna("Not Reported")
        .str.strip()
    )

    # -----------------------------
    # Age grouping
    # -----------------------------
    df["age_group"] = pd.cut(
        df["age"],
        bins=[17, 25, 35, 45, 55, 70, float("inf")],
        labels=[
            "18-25",
            "26-35",
            "36-45",
            "46-55",
            "56-70",
            "70+"
        ]
    )

    # -----------------------------
    # Confirmed logical outliers
    # -----------------------------
    outlier_mask = (
        (
            (df["age_group"] == "56-70") &
            (df["occupation"] == "Student")
        )
        |
        (df["age_group"] == "70+")
    )

    outlier_rows = int(outlier_mask.sum())

    df = df.loc[~outlier_mask].copy()

    # Store age_group as normal string
    df["age_group"] = df["age_group"].astype(str)

    stats = {
        "rows_raw": rows_raw,
        "duplicate_rows_removed": duplicate_rows,
        "logical_outliers_removed": outlier_rows,
        "rows_clean": len(df),
        "columns_clean": len(df.columns),
        "remaining_missing": {
            col: int(value)
            for col, value in df.isna().sum().items()
            if value > 0
        }
    }

    return df, stats


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-path",
        default="/opt/ml/processing/input/survey_results.csv"
    )

    parser.add_argument(
        "--output-dir",
        default="/opt/ml/processing/output"
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_csv(args.input_path)

    df_clean, stats = clean_data(df)

    output_csv = os.path.join(
        args.output_dir,
        "cleaned_survey_results.csv"
    )

    output_metadata = os.path.join(
        args.output_dir,
        "preprocessing_metadata.json"
    )

    df_clean.to_csv(
        output_csv,
        index=False
    )

    with open(output_metadata, "w") as f:
        json.dump(stats, f, indent=4)

    print(json.dumps(stats, indent=4))
    print(f"\nSaved cleaned data to: {output_csv}")


if __name__ == "__main__":
    main()
