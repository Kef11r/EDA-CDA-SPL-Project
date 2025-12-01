from pathlib import Path
from typing import Iterable, Optional
import pandas as pd
from .config import RAW_DATA_FILE, PROCESSED_DATA_FILE


def load_data(path: Path | str = RAW_DATA_FILE) -> pd.DataFrame:
    path = Path(path)
    return pd.read_csv(path)


def save_processed_data(df: pd.DataFrame,
                        path: Path | str = PROCESSED_DATA_FILE,
                        index: bool = False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)

def list_columns(df: pd.DataFrame) -> pd.DataFrame:
    info = pd.DataFrame({
        "column": df.columns,
        "dtype": [df[c].dtype for c in df.columns]
    })
    return info


def basic_info(df: pd.DataFrame) -> None:
    print("Shape:", df.shape)
    print("\nDtypes:")
    print(df.dtypes)
    print("\nMissing values per column:")
    print(df.isna().sum())


def numerical_summary(df: pd.DataFrame,
                      columns: Optional[Iterable[str]] = None) -> pd.DataFrame:
    if columns is None:
        num_df = df.select_dtypes(include="number")
    else:
        num_df = df[list(columns)]
    return num_df.describe().T

def categorical_summary(df: pd.DataFrame,
                        columns: Optional[Iterable[str]] = None,
                        top_n: int = 5) -> dict[str, pd.Series]:
    if columns is None:
        cat_df = df.select_dtypes(exclude="number")
    else:
        cat_df = df[list(columns)]
    result: dict[str, pd.Series] = {}
    for col in cat_df.columns:
        result[col] = cat_df[col].value_counts().head(top_n)
    return result

def correlation_matrix(df: pd.DataFrame, columns: Optional[Iterable[str]] = None) -> pd.DataFrame:
    if columns is None:
        num_df = df.select_dtypes(include="number")
    else:
        num_df = df[list(columns)]
    return num_df.corr(numeric_only=True)
