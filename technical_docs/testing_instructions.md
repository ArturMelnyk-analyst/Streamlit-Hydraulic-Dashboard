# Testing Instructions
## Hydraulic Condition Monitoring Dashboard

**Author:** Artur Melnyk

This document explains how to test the hydraulic condition monitoring application and verify that the local dashboard, saved models, inference artifacts, and downloadable outputs work correctly.

---

## 1. Purpose

The goal of testing is to confirm that the deployed inference pipeline works end to end:

- trained `.joblib` subsystem models load correctly
- feature-engineered input files are accepted by the application
- feature alignment is preserved using `artifacts/feature_index.json`
- predictions are generated for all subsystem targets
- downloadable CSV outputs match the displayed dashboard results

This is not a unit-test suite. It is a practical application-validation procedure for local and deployment-facing testing.

---

## 2. Files Used in Testing

### Required application files

```text
streamlit_dashboard_app.py
models/
artifacts/feature_index.json
artifacts/<target>_label_map.json
requirements.txt
runtime.txt
```

### Input and output folders

```text
data/testing_input/
data/testing_output/
data/testing_sample/
```

### Validation helper script

```text
generate_testing_outputs.py
```

---

## 3. What Is Being Tested

The testing workflow verifies five things:

### A. Application startup
The Streamlit app launches without import or environment errors.

### B. Input validation
The application accepts only feature-engineered inputs with the expected schema.

### C. Inference consistency
The app uses the same saved models and feature order as the training pipeline.

### D. Output generation
The app displays predictions, labels, and confidence scores for every subsystem.

### E. Result export
The app generates downloadable CSV outputs successfully.

---

## 4. Pre-Test Requirements

Before testing, confirm that:

- Python version matches the environment expected by `runtime.txt`
- all packages from `requirements.txt` are installed
- the `models/` folder contains all saved subsystem models
- the `artifacts/` folder contains `feature_index.json` and label maps
- the `data/testing_*` folders are available if reproducible test inputs are being used

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run streamlit_dashboard_app.py
```

---

## 5. Recommended Test Order

Run the tests in the following order:

1. Environment and startup test
2. Demo sample test
3. Uploaded CSV schema test
4. Prediction correctness test
5. Download/export test
6. Regression re-check after changes

This order helps isolate failures efficiently.

---

## 6. Test 1 — Environment and Startup

### Objective
Verify that the app starts successfully and loads the required environment.

### Steps
1. Open a terminal in the project root.
2. Run:

```bash
streamlit run streamlit_dashboard_app.py
```

3. Wait for the local Streamlit page to open.

### Expected result
- the app launches successfully
- no import errors appear
- no missing dependency errors appear
- the dashboard page renders correctly

### Failure indicates
- missing package in `requirements.txt`
- incorrect Python runtime
- broken imports
- environment mismatch between local and deployment setup

---

## 7. Test 2 — Demo Sample Validation

### Objective
Verify that the built-in demo sample can be loaded and processed correctly.

### Steps
1. Open the dashboard.
2. Select **Use demo sample**.
3. Wait for the app to load the test input.

### Expected result
- the success message appears
- input preview is shown
- the feature-engineered sample loads correctly
- schema validation passes

### What this proves
- test sample access works
- feature matrix shape is valid
- the app can process a known-good inference input

---

## 8. Test 3 — Uploaded CSV Validation

### Objective
Verify that user-uploaded feature-engineered CSV files are accepted only when their schema is correct.

### Steps
1. Select **Upload CSV**.
2. Upload a valid feature-engineered CSV file generated from the pipeline.
3. Observe dashboard behavior.

### Expected result
- the file uploads successfully
- schema validation passes
- prediction output becomes available

### Negative test
Also upload an intentionally incorrect file, for example:
- wrong columns
- missing columns
- reordered columns without alignment support
- non-feature-engineered raw input

### Expected negative result
- the app rejects the file or raises a clear validation error
- no invalid predictions are generated

### What this proves
- inference requires the correct feature contract
- the app does not silently accept broken inputs

---

## 9. Test 4 — Prediction Generation

### Objective
Verify that predictions are generated for all subsystem targets.

### Expected subsystem outputs
- Cooler condition
- Valve condition
- Pump leakage
- Accumulator pressure
- Stable / unstable flag

### Steps
1. Use a valid uploaded CSV or demo sample.
2. Wait for predictions to appear.

### Expected result
The dashboard shows:
- predicted encoded value
- decoded label
- confidence score or probability-style output
- one result per subsystem target

### What this proves
- models load correctly
- all five subsystem models are active
- inference artifacts are aligned properly
- label decoding works

---

## 10. Test 5 — Downloadable Output Validation

### Objective
Verify that the download buttons create correct output files.

### Steps
1. Run a successful prediction.
2. Click **Download predictions only (CSV)**.
3. Click **Download compact output (CSV)**.
4. Open the downloaded files locally.

### Expected result
- both files download successfully
- the exported predictions match the dashboard output
- labels and confidence values remain consistent
- no columns are missing unexpectedly

### What this proves
- output export logic works
- post-inference formatting is correct
- the dashboard supports operational handoff of results

---

## 11. Test 6 — Reproducible Validation with Helper Script

### Objective
Verify that known test samples and expected outputs can be reproduced.

### Script
```text
generate_testing_outputs.py
```

### Role of the script
This script is used to:
- generate testing inputs
- generate expected prediction outputs
- support lightweight regression checking after code or environment changes

### Recommended use
Run the script whenever:
- models are updated
- artifact files are replaced
- the app code changes
- dependencies change
- the deployment environment is rebuilt

### Expected result
The script regenerates testing artifacts that can be compared with dashboard behavior.

---

## 12. Pass / Fail Criteria

A test cycle is considered successful when:

- the dashboard launches successfully
- a valid demo sample loads
- valid uploaded CSV inputs are accepted
- invalid inputs are rejected or flagged
- predictions appear for all five subsystem targets
- download buttons create correct files
- regenerated expected outputs remain consistent with dashboard predictions

A test cycle fails when any one of these steps breaks.

---

## 13. Common Failure Modes

### Missing package error
Likely cause:
- dependency missing from `requirements.txt`

### App works locally but not in Streamlit Cloud
Likely cause:
- mismatch between local Python and `runtime.txt`
- outdated deployment environment
- package version mismatch

### File uploads but predictions fail
Likely cause:
- broken feature alignment
- mismatched `feature_index.json`
- incorrect engineered feature schema

### Labels are wrong but model outputs exist
Likely cause:
- broken or outdated label mapping files

### Download buttons fail
Likely cause:
- dashboard export logic issue
- output dataframe formatting issue

---

## 14. Deployment Re-Check After Dependency Updates

Whenever `requirements.txt` or `runtime.txt` changes:

1. commit the updated files
2. push to GitHub
3. allow Streamlit Cloud to rebuild the environment
4. rerun the same tests in the deployed app

This step is important because a locally working app may still fail in deployment if the cloud runtime differs from the local environment.

---

## 15. Testing Evidence

The following files and screenshots should be retained as validation evidence:

- dashboard landing page
- demo sample loaded successfully
- prediction results
- downloaded CSV output
- sample testing input and output files
- `requirements.txt`
- `runtime.txt`

These artifacts provide reproducible evidence that the dashboard, models, and export workflow function correctly.

---

## 16. Final Summary

The hydraulic dashboard should be treated as a deployable inference interface, not just a visual add-on.

A successful test process confirms that the system works from end to end:

```text
Saved models
    ↓
Inference artifacts
    ↓
Valid feature-engineered input
    ↓
Subsystem predictions
    ↓
Decoded labels
    ↓
Downloaded CSV outputs
```

That makes testing a core part of the project’s engineering quality, reproducibility, and deployment readiness.
