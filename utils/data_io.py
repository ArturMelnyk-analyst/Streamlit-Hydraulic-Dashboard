"""
utils/data_io.py — Hydraulic dataset loaders

Folders used (relative to repo root):
- data/raw/        : original .txt sensor files (+ profile.txt if you keep it there)
- data/metadata/   : non-sensor files (profile.txt, docs, features_index.csv, etc.)
- data/processed/  : saved merged tables (X_features.parquet, y_labels.parquet)

This module:
- loads 17 sensor files (whitespace or ';' delims)
- loads profile.txt from metadata/ OR raw/
- standardizes column names <SENSOR>_t1..tN
- builds (X, y) and saves them
- optionally writes metadata/features_index.csv
"""

from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd

# -------- paths --------
def _root_dir() -> Path:
    try:
        return Path(__file__).resolve().parents[1]  # .../hydraulic_dashboard
    except NameError:
        return Path.cwd().resolve()

ROOT_DIR = _root_dir()
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
META_DIR = DATA_DIR / "metadata"
PROC_DIR = DATA_DIR / "processed"

IGNORED_NON_SENSOR = {"profile.txt", "description.txt", "documentation.txt"}

def ensure_dirs() -> None:
    for d in (DATA_DIR, RAW_DIR, META_DIR, PROC_DIR):
        d.mkdir(parents=True, exist_ok=True)

# -------- internals --------
def _detect_sep_and_read(path: Path) -> Tuple[pd.DataFrame, str]:
    # try whitespace first
    try:
        df = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
        if df.shape[1] > 1:
            return df, "whitespace"
    except Exception:
        pass
    # fallback ';'
    df = pd.read_csv(path, sep=";", header=None, engine="python")
    return df, "semicolon"

def _sensor_code(path: Path) -> str:
    return path.stem.upper()

def _name_columns(df: pd.DataFrame, sensor_code: str) -> pd.DataFrame:
    df = df.copy()
    df.columns = [f"{sensor_code}_t{i+1}" for i in range(df.shape[1])]
    return df

def _ok(msg: str) -> None:
    print(f"✅ {msg}")

# -------- public API --------
def load_profile() -> pd.DataFrame:
    """
    Load labels (profile.txt) from data/metadata OR data/raw.
    Returns a 5-column DataFrame; column names are added later.
    """
    for p in (META_DIR / "profile.txt", RAW_DIR / "profile.txt"):
        if p.exists():
            df = pd.read_csv(p, sep=r"\s+", header=None, engine="python")
            _ok(f"Loaded profile.txt | shape={df.shape} | from={p.relative_to(ROOT_DIR)}")
            return df
    raise FileNotFoundError("profile.txt not found in data/metadata or data/raw")

def load_all_sensors() -> Tuple[Dict[str, pd.DataFrame], List[str]]:
    """
    Load all sensor .txt files from data/raw (excludes metadata/label files).
    Returns (dict of DataFrames by sensor code, ordered list of codes).
    """
    files = sorted(RAW_DIR.glob("*.txt"))
    sensor_files = [f for f in files if f.name not in IGNORED_NON_SENSOR]

    print("— Loading sensor files —")
    frames: Dict[str, pd.DataFrame] = {}
    order: List[str] = []
    for f in sensor_files:
        df, delim = _detect_sep_and_read(f)
        code = _sensor_code(f)
        df = _name_columns(df, code)
        print(f"   Loaded {f.name:<14} | shape={str(df.shape):<14} | delim={delim}")
        frames[code] = df
        order.append(code)

    print(f"\nTotal sensor files loaded: {len(sensor_files)}")
    skipped = [f.name for f in files if f.name in IGNORED_NON_SENSOR]
    if skipped:
        print(f"Skipped non-sensor files: {', '.join(skipped)}")
    return frames, order

def build_features_matrix(sensor_frames: Dict[str, pd.DataFrame],
                         sensor_order: List[str],
                         labels_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Concatenate sensor frames column-wise and prepare y with readable names.
    """
    X = pd.concat([sensor_frames[k] for k in sensor_order], axis=1)

    label_names = [
        "cooler_condition",
        "valve_condition",
        "internal_pump_leakage",
        "hydraulic_accumulator",
        "stable_flag",
    ]
    y = labels_df.copy()
    if y.shape[1] == 5:
        y.columns = label_names

    _ok(f"Built features matrix X | shape={X.shape}")
    _ok(f"Prepared labels y       | shape={y.shape} | columns={list(y.columns)}")
    return X, y

def save_processed(X: pd.DataFrame, y: pd.DataFrame) -> Tuple[Path, Path]:
    """
    Save X and y to data/processed as Parquet.
    These are the canonical frozen artifacts consumed by 02_eda, 03_feature_engineering, 04_modeling.

    Returns:
        (path_to_X, path_to_y)
    """
    PROC_DIR.mkdir(parents=True, exist_ok=True)

    # Canonical artifact names for downstream notebooks
    x_pq = PROC_DIR / "X_features.parquet"
    y_pq = PROC_DIR / "y_labels.parquet"

    try:
        X.to_parquet(x_pq, index=False)
        y.to_parquet(y_pq, index=False)
        _ok(f"Saved X -> {x_pq.relative_to(ROOT_DIR)}")
        _ok(f"Saved y -> {y_pq.relative_to(ROOT_DIR)}")
        return x_pq, y_pq

    except Exception as e:
        _ok(f"Parquet not available ({e}); saving CSV instead as fallback.")
        x_csv = PROC_DIR / "X_features.csv"
        y_csv = PROC_DIR / "y_labels.csv"
        X.to_csv(x_csv, index=False)
        y.to_csv(y_csv, index=False)
        _ok(f"Saved X -> {x_csv.relative_to(ROOT_DIR)}")
        _ok(f"Saved y -> {y_csv.relative_to(ROOT_DIR)}")
        return x_csv, y_csv

def write_features_index(X: pd.DataFrame) -> Path:
    """
    Generate a column dictionary for downstream analysis and modeling.

    Output columns:
        - column : exact column name in X
        - sensor : base sensor code (e.g. 'PS1', 'TS2', 'VS1', 'SE', ...)
        - group  : high-level functional group
                   ('pressure', 'flow', 'temperature', 'cooler', 'vibration', 'electrical', 'other')

    This file becomes data/metadata/features_index.csv
    and is consumed by 02_eda, 03_feature_engineering, and 04_modeling.
    """

    META_DIR.mkdir(parents=True, exist_ok=True)

    sensor_groups = {
        "pressure":   [f"PS{i}" for i in range(1, 7)],
        "flow":       ["FS1", "FS2"],
        "temperature":[f"TS{i}" for i in range(1, 5)],
        "cooler":     ["CE", "CP"],
        "vibration":  ["VS1"],
        "electrical": ["SE", "EPS1"],
    }

    rows = []
    for col in X.columns:
        sensor = col.split("_", 1)[0]  # e.g. 'PS1' from 'PS1_t3'

        # map sensor -> functional group
        group = next(
            (g for g, sensors in sensor_groups.items() if sensor in sensors),
            "other"
        )

        rows.append({
            "column": col,
            "sensor": sensor,
            "group": group
        })

    df = pd.DataFrame(rows)

    out = META_DIR / "features_index.csv"
    df.to_csv(out, index=False)

    _ok(f"Wrote metadata -> {out.relative_to(ROOT_DIR)}; rows={len(df)}")
    return out