import pandas as pd
import numpy as np
import re
from typing import Optional, List, Union

class DC:
    def __init__(self, data: Union[str, pd.DataFrame], **read_csv_kwargs):
        if isinstance(data, str):
            self.data = pd.read_csv(data, **read_csv_kwargs)
            self.source_file = data
        elif isinstance(data, pd.DataFrame):
            self.data = data.copy()
            self.source_file = None
        else:
            raise ValueError("data must be a CSV path or a pandas DataFrame")

    def info(self) -> "DC":
        print("Columns:", list(self.data.columns))
        print("Shape:", self.data.shape)
        print("\nDtypes:\n", self.data.dtypes)
        print(f"\nMemory: {self.data.memory_usage(deep=True).sum()/1024**2:.2f} MB")
        return self

    def Nan_view(self, show_rows: bool = False) -> "DC":
        counts = self.data.isnull().sum()
        pct = (counts / len(self.data) * 100).round(2)
        report = pd.DataFrame({"NaN Count": counts, "NaN %": pct}).sort_values("NaN Count", ascending=False)
        print("=== NaN Report ===")
        print(report[report["NaN Count"] > 0] if report["NaN Count"].sum() else "No NaNs.")
        if show_rows and self.data.isnull().any(axis=1).any():
            null_rows = self.data[self.data.isnull().any(axis=1)]
            print("\nRows with any NaN:", len(null_rows))
            print(null_rows.head(10))
        return self

    def Nan_drop(self, axis: int = 0, thresh: Optional[int] = None, subset: Optional[List[str]] = None) -> "DC":
        before = self.data.shape
        self.data = self.data.dropna(axis=axis, thresh=thresh, subset=subset)
        after = self.data.shape
        if axis == 0:
            print(f"Dropped {before[0] - after[0]} rows")
        else:
            print(f"Dropped {before[1] - after[1]} columns")
        return self

    def fill_NaN(self, column: Optional[str] = None, method: str = "auto", fill_value=None) -> "DC":
        def _fill_col(col: str):
            missing = self.data[col].isna().sum()
            if missing == 0:
                return
            if method == "auto":
                if pd.api.types.is_numeric_dtype(self.data[col]):
                    val = self.data[col].mean()
                    self.data[col] = self.data[col].fillna(val)
                    print(f"[{col}] filled {missing} with mean={val:.4f}")
                else:
                    mode = self.data[col].mode()
                    val = mode.iloc[0] if not mode.empty else "NOT GIVEN"
                    self.data[col] = self.data[col].fillna(val)
                    print(f"[{col}] filled {missing} with mode='{val}'")
            elif method == "mean":
                val = self.data[col].mean()
                self.data[col] = self.data[col].fillna(val)
                print(f"[{col}] filled {missing} with mean={val:.4f}")
            elif method == "median":
                val = self.data[col].median()
                self.data[col] = self.data[col].fillna(val)
                print(f"[{col}] filled {missing} with median={val:.4f}")
            elif method == "mode":
                mode = self.data[col].mode()
                val = mode.iloc[0] if not mode.empty else "NOT GIVEN"
                self.data[col] = self.data[col].fillna(val)
                print(f"[{col}] filled {missing} with mode='{val}'")
            elif method == "ffill":
                self.data[col] = self.data[col].fillna(method="ffill")
                print(f"[{col}] forward-filled {missing}")
            elif method == "bfill":
                self.data[col] = self.data[col].fillna(method="bfill")
                print(f"[{col}] backward-filled {missing}")
            elif method == "custom":
                if fill_value is None:
                    raise ValueError("fill_value required when method='custom'")
                self.data[col] = self.data[col].fillna(fill_value)
                print(f"[{col}] filled {missing} with custom value '{fill_value}'")
            else:
                raise ValueError("Invalid method")

        if column:
            if column not in self.data.columns:
                raise ValueError(f"Column '{column}' not in DataFrame")
            _fill_col(column)
        else:
            for col in self.data.columns:
                _fill_col(col)
        return self

    def full_clean(self, column: str, drop_dupes: bool = True, lower: bool = False) -> "DC":
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' not found.")

        s = self.data[column].astype("string")
        s = s.str.strip()
        if lower:
            s = s.str.lower()

        def _rm_dup_words(text: Optional[str]) -> Optional[str]:
            if text is None or pd.isna(text):
                return text
            return re.sub(r'\b(\w+)(?:\s+\1\b)+', r'\1', text, flags=re.IGNORECASE)

        s = s.apply(_rm_dup_words)
        s = s.replace(r"https?://\S+|www\.\S+", "", regex=True)
        self.data[column] = s

        if drop_dupes:
            before = len(self.data)
            self.data = self.data.drop_duplicates()
            print(f"Dropped {before - len(self.data)} duplicate rows")

        return self

    def standardize_column_names(self) -> "DC":
        before = list(self.data.columns)
        self.data.columns = [c.strip().lower().replace(" ", "_").replace("-", "_") for c in self.data.columns]
        after = list(self.data.columns)
        changes = [f"'{b}' -> '{a}'" for b, a in zip(before, after) if b != a]
        if changes:
            print("Renamed:", "; ".join(changes))
        return self

    def export(self, path: str, **to_csv_kwargs) -> "DC":
        self.data.to_csv(path, index=False, **to_csv_kwargs)
        print(f"Saved to {path} | shape={self.data.shape}")
        return self

    def get(self) -> pd.DataFrame:
        return self.data.copy()