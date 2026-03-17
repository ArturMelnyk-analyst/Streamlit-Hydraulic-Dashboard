from pathlib import Path
import json
import joblib
import pandas as pd

BASE_DIR = Path(".").resolve()

DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
ARTIFACTS_DIR = BASE_DIR / "artifacts"

TEST_INPUT_DIR = DATA_DIR / "testing_input"
TEST_OUTPUT_DIR = DATA_DIR / "testing_output"

TARGET_TO_MODEL_FILE = {
    "Cooler_Condition": "cooler_model.joblib",
    "Valve_Condition": "valve_model.joblib",
    "Pump_Leakage": "pump_model.joblib",
    "Accumulator_Pressure": "accumulator_model.joblib",
    "Stable_Flag": "stable_model.joblib",
}

TARGET_TO_LABEL_MAP_FILE = {
    "Cooler_Condition": "Cooler_Condition_label_map.json",
    "Valve_Condition": "Valve_Condition_label_map.json",
    "Pump_Leakage": "Pump_Leakage_label_map.json",
    "Accumulator_Pressure": "Accumulator_Pressure_label_map.json",
    "Stable_Flag": "Stable_Flag_label_map.json",
}

def read_json_if_exists(path: Path):
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

def normalize_label_map(raw_map):
    if raw_map is None:
        return {}
    out = {}
    for k, v in raw_map.items():
        try:
            out[int(k)] = str(v)
        except Exception:
            out[str(k)] = str(v)
    return out

def apply_label_map(series: pd.Series, label_map: dict) -> pd.Series:
    if not label_map:
        return series.astype(str)
    return series.map(lambda x: label_map.get(x, label_map.get(str(x), str(x))))

models = {
    target: joblib.load(MODELS_DIR / filename)
    for target, filename in TARGET_TO_MODEL_FILE.items()
}

label_maps = {
    target: normalize_label_map(read_json_if_exists(ARTIFACTS_DIR / map_file))
    for target, map_file in TARGET_TO_LABEL_MAP_FILE.items()
}

def make_expected_output(input_csv: str, output_csv: str):
    input_df = pd.read_csv(TEST_INPUT_DIR / input_csv)
    result_df = pd.DataFrame(index=input_df.index)

    for target, model in models.items():
        pred = model.predict(input_df)
        pred_series = pd.Series(pred, index=input_df.index)

        result_df[f"{target}_pred"] = pred_series
        result_df[f"{target}_label"] = apply_label_map(pred_series, label_maps.get(target, {}))

        if hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(input_df)
                result_df[f"{target}_confidence"] = proba.max(axis=1)
            except Exception:
                pass

    result_df.to_csv(TEST_OUTPUT_DIR / output_csv, index=False)
    print(f"Saved: {TEST_OUTPUT_DIR / output_csv}")

make_expected_output("single_case_input.csv", "single_case_expected_output.csv")
make_expected_output("batch_input.csv", "batch_expected_output.csv")