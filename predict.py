"""
Inference script for Track 3: Semiconductor Wafer Quality & Fault Detection.
Usage:
    python predict.py input.csv predictions.csv
Input may contain timestamp and sensor columns. The target column is not required.
"""
import sys
import joblib
import pandas as pd

def predict_dataframe(input_df, bundle_path="model.joblib"):
    bundle = joblib.load(bundle_path)
    sensor_columns = bundle["sensor_columns"]
    model = bundle["model"]
    threshold = bundle["threshold"]

    # Align incoming schema to training features. Missing expected sensors become NaN
    # and are handled by the saved imputer; extra columns are ignored.
    X = input_df.copy()
    for col in sensor_columns:
        if col not in X.columns:
            X[col] = float("nan")
    X = X[sensor_columns].apply(pd.to_numeric, errors="coerce")

    probabilities = model.predict_proba(X)[:, 1]
    labels = (probabilities >= threshold).astype(int)

    result = pd.DataFrame({
        "predicted_target": labels,
        "defect_probability": probabilities
    })
    if "timestamp" in input_df.columns:
        result.insert(0, "timestamp", input_df["timestamp"].values)
    return result

if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("Usage: python predict.py input.csv predictions.csv")
    input_path, output_path = sys.argv[1], sys.argv[2]
    input_df = pd.read_csv(input_path)
    predictions = predict_dataframe(input_df)
    predictions.to_csv(output_path, index=False)
    print(f"Saved {len(predictions)} predictions to {output_path}")
