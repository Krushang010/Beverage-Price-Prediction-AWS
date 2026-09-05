
import os
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


INPUT_DIR = "/opt/ml/processing/input"
TRAIN_DIR = "/opt/ml/processing/train"
TEST_DIR = "/opt/ml/processing/test"
METADATA_DIR = "/opt/ml/processing/metadata"

TARGET = "price_range"
TEST_SIZE = 0.25
RANDOM_STATE = 42


def main():

    input_files = list(Path(INPUT_DIR).rglob("*.csv"))

    if not input_files:
        raise FileNotFoundError(
            f"No CSV found under {INPUT_DIR}"
        )

    input_file = input_files[0]

    print("Input:", input_file)

    df = pd.read_csv(input_file)

    if TARGET not in df.columns:
        raise ValueError(
            f"Target column '{TARGET}' not found."
        )

    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df[TARGET],
    )

    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(TEST_DIR, exist_ok=True)
    os.makedirs(METADATA_DIR, exist_ok=True)

    train_path = os.path.join(
        TRAIN_DIR,
        "train.csv"
    )

    test_path = os.path.join(
        TEST_DIR,
        "test.csv"
    )

    train_df.to_csv(
        train_path,
        index=False
    )

    test_df.to_csv(
        test_path,
        index=False
    )

    metadata = {
        "total_rows": int(len(df)),
        "training_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "stratified_by": TARGET,
    }

    with open(
        os.path.join(
            METADATA_DIR,
            "split_metadata.json"
        ),
        "w"
    ) as f:
        json.dump(metadata, f, indent=4)

    print(json.dumps(metadata, indent=4))
    print("Train:", train_path)
    print("Test :", test_path)


if __name__ == "__main__":
    main()
