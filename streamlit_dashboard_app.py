# streamlit_dashboard_app.py

import streamlit as st
import pandas as pd
import joblib
import os

# === Load Trained Models ===
@st.cache_resource
def load_models():
    return {
        'Cooler_Condition': joblib.load('models/cooler_model.joblib'),
        'Valve_Condition': joblib.load('models/valve_model.joblib'),
        'Pump_Leakage': joblib.load('models/pump_model.joblib'),
        'Accumulator_Pressure': joblib.load('models/accumulator_model.joblib'),
        'Stable_Flag': joblib.load('models/stable_model.joblib')
    }

models = load_models()

# === App Title ===
st.title("Hydraulic System Failure Predictor")
st.markdown("Upload a CSV with sensor statistics to get condition predictions")

# === Upload Input CSV ===
uploaded_file = st.file_uploader("Upload your feature-engineered CSV", type=["csv"])

if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
    st.write("### Input Data Preview:")
    st.dataframe(input_df.head())

    # === Make Predictions ===
    st.write("### Predictions:")
    results = {}
    for label, model in models.items():
        pred = model.predict(input_df)
        results[label] = pred
    result_df = pd.DataFrame(results)
    st.dataframe(result_df)

    # === Download Button ===
    csv = result_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Predictions as CSV", csv, "predictions.csv", "text/csv")
else:
    st.info("Please upload a CSV file to begin.")
