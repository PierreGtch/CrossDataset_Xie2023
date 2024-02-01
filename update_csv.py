from pathlib import Path

import pandas as pd
from finetuning import csv_columns

default_values = dict(
    ref_channel="average",
)


def main(dir):
    for file in Path(dir).glob("*.csv"):
        df = pd.read_csv(file)
        print(f"{file}")
        for k, v in default_values.items():
            if k not in df.columns:
                df[k] = v
                print(f"- Added column {k}={v}")
        assert set(df.columns) == set(csv_columns)
        df = df[csv_columns]
        df.to_csv(file, index=False, mode="w", header=True)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--dir", type=str, default="./results")

    args = parser.parse_args()

    main(args.dir)
