"""pipeline/preprocess.py — Data loading utilities"""
import re
import pandas as pd

TEXT_COLS  = ["text","complaint","Complaint","ticket","Consumer complaint narrative","narrative","description"]
LABEL_COLS = ["label","category","Product","product","topic"]


def load_data(path: str) -> pd.DataFrame:
    df  = pd.read_csv(path, low_memory=False)
    col = next((c for c in TEXT_COLS if c in df.columns), None)
    if col is None:
        raise KeyError(f"No text column found. Tried: {TEXT_COLS}")
    df  = df[[col]].dropna()
    df.columns = ["ticket"]
    df["ticket"] = df["ticket"].astype(str).str.strip()
    return df


def load_labeled_data(path: str) -> pd.DataFrame:
    df    = pd.read_csv(path, low_memory=False)
    tcol  = next((c for c in TEXT_COLS  if c in df.columns), None)
    lcol  = next((c for c in LABEL_COLS if c in df.columns), None)
    if tcol is None: raise KeyError(f"No text column.  Tried: {TEXT_COLS}")
    if lcol is None: raise KeyError(f"No label column. Tried: {LABEL_COLS}")
    df    = df[[tcol, lcol]].dropna()
    df.columns = ["ticket", "label"]
    df["ticket"] = df["ticket"].astype(str).str.strip()
    df["label"]  = df["label"].astype(str).str.strip()
    return df


def clean_text(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s.,!?'-]", "", text)
    return text
