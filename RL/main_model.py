# file: preprocess_rl_data.py

import glob
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler

def load_all_data(input_dir: str, pattern: str = "*.csv", testing: bool = False) -> pd.DataFrame:
    """
    Read all CSVs matching `pattern` under `input_dir` into one DataFrame.
    Parses 'DATE' as datetime.
    
    Args:
        input_dir: Directory containing CSV files
        pattern: Glob pattern to match files
        testing: If True, only read first 20000 lines from each file
    """
    files = glob.glob(str(Path(input_dir) / pattern))
    if not files:
        raise FileNotFoundError(f"No files matched {input_dir}/{pattern}")
    dfs = []
    for f in files:
        if testing:
            df = pd.read_csv(f, parse_dates=["DATE"], nrows=20000)
        else:
            df = pd.read_csv(f, parse_dates=["DATE"])
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def drop_extreme_rows(
    df: pd.DataFrame, numeric_cols: list[str], lower_q: float = 0.001, upper_q: float = 0.999
) -> pd.DataFrame:
    """
    Remove any row where *any* numeric column is below its `lower_q`-quantile
    or above its `upper_q`-quantile.
    """
    # compute bounds
    lowers = df[numeric_cols].quantile(lower_q)
    uppers = df[numeric_cols].quantile(upper_q)

    # build mask
    mask = np.ones(len(df), dtype=bool)
    for col in numeric_cols:
        mask &= df[col].between(lowers[col], uppers[col])
    return df[mask].reset_index(drop=True)


def normalize_numeric(
    df: pd.DataFrame, numeric_cols: list[str]
) -> tuple[pd.DataFrame, StandardScaler]:
    """
    Fit a StandardScaler on the numeric columns and transform them.
    Returns (df_transformed, fitted_scaler).
    """
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df, scaler


def main():
    import argparse

    p = argparse.ArgumentParser(
        description="Load CSVs, trim extreme 0.1% outliers, normalize numeric for RL"
    )
    p.add_argument(
        "--input_dir",
        "-i",
        required=True,
        help="Directory containing raw CSV files",
    )
    p.add_argument(
        "--output_csv",
        "-o",
        help="Path for the cleaned output CSV",
    )
    p.add_argument(
        "--columns",
        "-c",
        nargs="+",
        required=True,
        help=(
            "List of columns to keep (must include 'DATE' and 'SYM_ROOT' plus your numeric features). "
            "E.g.: --columns DATE SYM_ROOT total_vol CPrc ivol_t"
        ),
    )
    p.add_argument(
        "--testing",
        action="store_true",
        help="If set, limit data loading to 20000 rows for testing"
    )
    args = p.parse_args()

    # 1. load
    df = load_all_data(args.input_dir, maxrows=20000 if args.testing else None)

    # 2. select only the requested columns
    missing = set(args.columns) - set(df.columns)
    if missing:
        raise KeyError(f"These requested columns are not in the data: {missing}")
    df = df[args.columns].copy()

    # 3. identify numeric columns (exclude DATE, SYM_ROOT)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # 4. drop extreme 0.1% outliers
    df = drop_extreme_rows(df, numeric_cols, lower_q=0.001, upper_q=0.999)

    # 5. normalize numeric features
    df, scaler = normalize_numeric(df, numeric_cols)

    # # 6. write out
    # out_path = Path(args.output_csv)
    # out_path.parent.mkdir(parents=True, exist_ok=True)
    # df.to_csv(out_path, index=False)

    # print(f"✔ Saved cleaned data ({df.shape[0]} rows × {df.shape[1]} cols) → {out_path}")


if __name__ == "__main__":
    main()
