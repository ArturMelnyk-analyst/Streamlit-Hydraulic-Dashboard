from __future__ import annotations

from pathlib import Path
import json
import warnings

import joblib
import pandas as pd
import streamlit as st


# ============================================================
# 0. Project paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
ARTIFACTS_DIR = BASE_DIR / "artifacts"

TEST_INPUT_DIR = DATA_DIR / "testing_input"
TEST_OUTPUT_DIR = DATA_DIR / "testing_output"
TEST_SAMPLE_DIR = DATA_DIR / "testing_sample"

warnings.filterwarnings("ignore")


# ============================================================
# 1. Config
# ============================================================

st.set_page_config(
    page_title="Hydraulic Condition Monitoring Dashboard",
    page_icon="🛠️",
    layout="wide"
)

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


# ============================================================
# 2. UI translations
# ============================================================

TRANSLATIONS = {
    "English": {
        "sidebar_title": "Settings",
        "language_label": "Language / Sprache",
        "title": "Hydraulic Condition Monitoring Dashboard",
        "intro": """
Upload a **feature-engineered CSV** to generate subsystem condition predictions.

This app predicts:
- Cooler condition
- Valve condition
- Pump leakage
- Accumulator pressure condition
- Stable / unstable state
""",
        "engineering_note_title": "Engineering note",
        "engineering_note": (
            "Only visible UI text is translated. Technical target names, model artifacts, "
            "feature names, and SHAP outputs remain in English to preserve consistency between "
            "training and inference."
        ),
        "input_source": "Input Source",
        "choose_input_method": "Choose input method",
        "upload_csv": "Upload CSV",
        "use_demo_sample": "Use demo sample",
        "demo_missing": "No demo sample found at data/testing_sample/demo_case.csv",
        "demo_loaded": "Loaded demo sample successfully.",
        "upload_prompt": "Upload your feature-engineered CSV",
        "csv_read_error": "Could not read the uploaded CSV.",
        "start_info": "Choose a demo sample or upload a CSV file to begin.",
        "empty_csv": "The provided CSV is empty.",
        "input_preview": "Input Preview",
        "rows": "Rows",
        "columns": "Columns",
        "preview_first_cols": "Preview of first 20 columns only:",
        "schema_warning": (
            "No canonical feature schema artifact was found. "
            "Validation is limited, so the app assumes your CSV already matches the trained model input."
        ),
        "missing_cols_error": "The uploaded file is missing required feature columns.",
        "extra_cols_warning": "The uploaded file contains extra columns. They will be ignored.",
        "validation_passed": "Input schema validation passed.",
        "predictions": "Predictions",
        "download_results": "Download Results",
        "download_predictions_only": "Download predictions only (CSV)",
        "download_compact": "Download compact output (CSV)",
        "prediction_failed": "Prediction failed.",
        "asset_load_failed": "Failed to load application assets.",
    },
    "Deutsch": {
        "sidebar_title": "Einstellungen",
        "language_label": "Sprache / Language",
        "title": "Dashboard zur Überwachung hydraulischer Systeme",
        "intro": """
Laden Sie eine **feature-engineerte CSV-Datei** hoch, um Zustandsvorhersagen für einzelne hydraulische Subsysteme zu erzeugen.

Diese App sagt vorher:
- Zustand des Kühlsystems
- Zustand des Ventils
- Pumpenleckage
- Zustand des Akkumulatordrucks
- Stabiler / instabiler Systemzustand
""",
        "engineering_note_title": "Technischer Hinweis",
        "engineering_note": (
            "Nur sichtbare Texte der Benutzeroberfläche werden übersetzt. Technische Zielnamen, "
            "Modellartefakte, Feature-Namen und SHAP-Ausgaben bleiben auf Englisch, damit Training "
            "und Inferenz konsistent bleiben."
        ),
        "input_source": "Eingabequelle",
        "choose_input_method": "Eingabemethode auswählen",
        "upload_csv": "CSV hochladen",
        "use_demo_sample": "Demo-Beispiel verwenden",
        "demo_missing": "Kein Demo-Beispiel unter data/testing_sample/demo_case.csv gefunden.",
        "demo_loaded": "Demo-Beispiel erfolgreich geladen.",
        "upload_prompt": "Feature-engineerte CSV-Datei hochladen",
        "csv_read_error": "Die hochgeladene CSV-Datei konnte nicht gelesen werden.",
        "start_info": "Wählen Sie ein Demo-Beispiel oder laden Sie eine CSV-Datei hoch, um zu beginnen.",
        "empty_csv": "Die bereitgestellte CSV-Datei ist leer.",
        "input_preview": "Eingabevorschau",
        "rows": "Zeilen",
        "columns": "Spalten",
        "preview_first_cols": "Vorschau der ersten 20 Spalten:",
        "schema_warning": (
            "Kein kanonisches Feature-Schema-Artefakt wurde gefunden. "
            "Die Validierung ist eingeschränkt, daher nimmt die App an, dass die CSV-Datei bereits "
            "zur trainierten Modelleingabe passt."
        ),
        "missing_cols_error": "In der hochgeladenen Datei fehlen erforderliche Feature-Spalten.",
        "extra_cols_warning": "Die hochgeladene Datei enthält zusätzliche Spalten. Diese werden ignoriert.",
        "validation_passed": "Validierung des Eingabeschemas erfolgreich.",
        "predictions": "Vorhersagen",
        "download_results": "Ergebnisse herunterladen",
        "download_predictions_only": "Nur Vorhersagen herunterladen (CSV)",
        "download_compact": "Kompakte Ausgabe herunterladen (CSV)",
        "prediction_failed": "Vorhersage fehlgeschlagen.",
        "asset_load_failed": "Anwendungsartefakte konnten nicht geladen werden.",
    },
}


# ============================================================
# 3. Helpers
# ============================================================

def read_json_if_exists(path: Path) -> dict | None:
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def normalize_label_map(raw_map: dict | None) -> dict:
    if raw_map is None:
        return {}

    normalized = {}
    for k, v in raw_map.items():
        try:
            normalized[int(k)] = str(v)
        except Exception:
            normalized[str(k)] = str(v)
    return normalized


def apply_label_map(series: pd.Series, label_map: dict) -> pd.Series:
    if not label_map:
        return series.astype(str)
    return series.map(lambda x: label_map.get(x, label_map.get(str(x), str(x))))


def load_model_with_compat(model_path: Path):
    try:
        return joblib.load(model_path)

    except AttributeError as e:
        err_msg = str(e)

        if "_RemainderColsList" in err_msg:
            try:
                import sklearn.compose._column_transformer as ct_mod

                if not hasattr(ct_mod, "_RemainderColsList"):
                    class _RemainderColsList(list):
                        pass

                    ct_mod._RemainderColsList = _RemainderColsList

                return joblib.load(model_path)

            except Exception as inner_e:
                raise RuntimeError(
                    f"Failed to load model '{model_path.name}' even after compatibility patch."
                ) from inner_e

        raise RuntimeError(
            f"Failed to load model '{model_path.name}'."
        ) from e


def load_expected_feature_list() -> list[str] | None:
    """
    Try to load canonical feature order.

    Supported shapes:
        - ["f1", "f2", ...]
        - [{"feature": "f1"}, ...]
        - {"features": [...]}
        - {"feature_order": [...]}

    Fallback:
        data/testing_sample/demo_case.csv
    """
    feature_index_path = ARTIFACTS_DIR / "feature_index.json"
    feature_index = read_json_if_exists(feature_index_path)

    if feature_index is not None:
        if isinstance(feature_index, list):
            if all(isinstance(x, str) for x in feature_index):
                return feature_index
            if all(isinstance(x, dict) and "feature" in x for x in feature_index):
                return [row["feature"] for row in feature_index]

        if isinstance(feature_index, dict):
            if "features" in feature_index and isinstance(feature_index["features"], list):
                return feature_index["features"]

            if "feature_order" in feature_index and isinstance(feature_index["feature_order"], list):
                return feature_index["feature_order"]

            candidate_keys = list(feature_index.keys())
            wrapper_keys = {"features", "feature_order", "metadata", "description"}

            if candidate_keys and not any(k in wrapper_keys for k in candidate_keys):
                return candidate_keys

    demo_path = TEST_SAMPLE_DIR / "demo_case.csv"
    if demo_path.exists():
        return pd.read_csv(demo_path, nrows=1).columns.tolist()

    return None


def validate_input_schema(input_df: pd.DataFrame, expected_features: list[str] | None):
    """
    Returns:
        is_valid, validated_df, missing_cols, extra_cols
    """
    if expected_features is None:
        return True, input_df.copy(), [], []

    input_cols = input_df.columns.tolist()
    expected_set = set(expected_features)
    input_set = set(input_cols)

    missing_cols = [c for c in expected_features if c not in input_set]
    extra_cols = [c for c in input_cols if c not in expected_set]

    if missing_cols:
        return False, None, missing_cols, extra_cols

    validated_df = input_df.loc[:, expected_features].copy()
    return True, validated_df, missing_cols, extra_cols


def load_demo_sample() -> pd.DataFrame | None:
    demo_path = TEST_SAMPLE_DIR / "demo_case.csv"
    if demo_path.exists():
        return pd.read_csv(demo_path)
    return None


def build_prediction_table(input_df: pd.DataFrame, models: dict, label_maps: dict) -> pd.DataFrame:
    results = pd.DataFrame(index=input_df.index)

    for target, model in models.items():
        pred = model.predict(input_df)
        pred_series = pd.Series(pred, index=input_df.index)

        results[f"{target}_pred"] = pred_series
        results[f"{target}_label"] = apply_label_map(pred_series, label_maps.get(target, {}))

        if hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(input_df)
                results[f"{target}_confidence"] = proba.max(axis=1)
            except Exception:
                pass

    return results


@st.cache_resource
def load_models() -> dict:
    models = {}
    missing = []

    for target, file_name in TARGET_TO_MODEL_FILE.items():
        model_path = MODELS_DIR / file_name
        if not model_path.exists():
            missing.append(file_name)
        else:
            models[target] = load_model_with_compat(model_path)

    if missing:
        raise FileNotFoundError("Missing model files: " + ", ".join(missing))

    return models


@st.cache_data
def load_label_maps() -> dict:
    label_maps = {}
    for target, map_file in TARGET_TO_LABEL_MAP_FILE.items():
        raw_map = read_json_if_exists(ARTIFACTS_DIR / map_file)
        label_maps[target] = normalize_label_map(raw_map)
    return label_maps


# ============================================================
# 4. Language selection
# ============================================================

language = st.sidebar.selectbox(
    "Language / Sprache",
    ["English", "Deutsch"],
    index=0
)
t = TRANSLATIONS[language]

st.sidebar.title(t["sidebar_title"])
st.sidebar.markdown("---")
st.sidebar.caption(t["engineering_note"])


# ============================================================
# 5. Load assets
# ============================================================

try:
    models = load_models()
    label_maps = load_label_maps()
    expected_features = load_expected_feature_list()
except Exception as e:
    st.error(t["asset_load_failed"])
    st.exception(e)
    st.stop()


# ============================================================
# 6. UI
# ============================================================

st.title(t["title"])
st.markdown(t["intro"])

with st.expander(t["engineering_note_title"], expanded=False):
    st.write(t["engineering_note"])

st.subheader(t["input_source"])

input_mode = st.radio(
    t["choose_input_method"],
    [t["upload_csv"], t["use_demo_sample"]],
    horizontal=True
)

input_df = None

if input_mode == t["use_demo_sample"]:
    demo_df = load_demo_sample()
    if demo_df is None:
        st.warning(t["demo_missing"])
    else:
        input_df = demo_df
        st.success(t["demo_loaded"])
else:
    uploaded_file = st.file_uploader(
        t["upload_prompt"],
        type=["csv"]
    )

    if uploaded_file is not None:
        try:
            input_df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(t["csv_read_error"])
            st.exception(e)
            st.stop()


# ============================================================
# 7. Validation + prediction
# ============================================================

if input_df is None:
    st.info(t["start_info"])
    st.stop()

if input_df.empty:
    st.error(t["empty_csv"])
    st.stop()

st.subheader(t["input_preview"])
st.write(f"{t['rows']}: {input_df.shape[0]}")
st.write(f"{t['columns']}: {input_df.shape[1]}")

preview_cols = input_df.columns[:20]
st.write(t["preview_first_cols"])
st.dataframe(input_df.loc[:, preview_cols].head(), use_container_width=True)

is_valid, validated_df, missing_cols, extra_cols = validate_input_schema(
    input_df=input_df,
    expected_features=expected_features
)

if expected_features is None:
    st.warning(t["schema_warning"])

if missing_cols:
    st.error(t["missing_cols_error"])
    st.code("\n".join(missing_cols[:100]))
    st.stop()

if extra_cols:
    st.warning(t["extra_cols_warning"])
    st.code("\n".join(extra_cols[:100]))

st.success(t["validation_passed"])

try:
    prediction_df = build_prediction_table(validated_df, models, label_maps)

    st.subheader(t["predictions"])
    st.dataframe(prediction_df, use_container_width=True)

    st.subheader(t["download_results"])

    # 1. Predictions only (recommended)
    prediction_csv = prediction_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label=t["download_predictions_only"],
        data=prediction_csv,
        file_name="hydraulic_predictions_only.csv",
        mime="text/csv"
    )

    # 2. Compact joined output
    reference_cols = input_df.columns[:10]
    compact_output_df = pd.concat(
        [
            input_df.loc[:, reference_cols].reset_index(drop=True),
            prediction_df.reset_index(drop=True)
        ],
        axis=1
    )

    compact_csv = compact_output_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label=t["download_compact"],
        data=compact_csv,
        file_name="hydraulic_predictions_compact.csv",
        mime="text/csv"
    )

except Exception as e:
    st.error(t["prediction_failed"])
    st.exception(e)
