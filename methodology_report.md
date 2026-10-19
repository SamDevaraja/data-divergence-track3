# Methodology Summary — Track 3: Semiconductor Wafer Quality & Fault Detection

## 1. Objective and dataset
The objective is binary classification of semiconductor wafers: `target=0` represents a passing wafer and `target=1` a defective wafer. The uploaded advanced-track dataset contains **1,253 observations**, **590 sensor features**, a timestamp, and the target. There are **1,170 passing wafers** and **83 defective wafers**, giving a defective rate of **6.62%**. This imbalance makes accuracy alone unsuitable as the main metric.

The sensor matrix contains **33,940 missing cells (4.59%)**. The uploaded advanced file has `sensor_0` through `sensor_589` (590 sensors), whereas the supplied problem description says `sensor_0` through `sensor_590` (591 sensors). The pipeline uses the actual uploaded schema and the inference script can align missing/extra columns.

## 2. Data preparation and modelling
The timestamp is retained for traceability but excluded from the baseline classifier. Sensor columns are converted to numeric values, with invalid values treated as missing. Median imputation is applied inside a scikit-learn pipeline, followed by removal of zero-variance features. The final classifier is an ExtraTrees ensemble with class-balanced weights. This approach is suitable as a baseline for nonlinear interactions in high-dimensional sensor data and avoids fitting the preprocessing on the validation partition.

The final pipeline was trained on all labelled rows and saved as `model.joblib`. `predict.py` loads the saved pipeline, aligns the incoming dataframe to the training sensor schema, returns binary predictions and defect probabilities, and does not require a `target` column in inference data.

## 3. Validation approach and results
A stratified 75/25 train/validation split was created with `random_state=42`, preserving the minority class ratio. A 5-fold stratified out-of-fold prediction procedure on the training partition selected the decision threshold; the validation partition was not used to select that threshold. The reported scores are local estimates only.

| Metric | Local validation result |
|---|---:|
| PR-AUC (Average Precision) | 0.1973 |
| Balanced Accuracy | 0.6164 |
| Defective-class precision | 0.0936 |
| Defective-class recall | 0.7619 |
| Defective-class F1 | 0.1667 |
| OOF-selected threshold | 0.052 |

At the selected threshold, the validation confusion matrix (TN, FP, FN, TP) is **[138, 155, 5, 16]**. The relatively low precision indicates false alarms remain a key limitation. Threshold selection trades precision against recall and should be selected based on the operational cost of missed defects versus unnecessary inspection. PR-AUC measures ranking quality across thresholds, while Balanced Accuracy gives equal weight to both classes.

## 4. Practical implications and interpretability
A useful production system would flag wafers for additional inspection before downstream packaging, potentially reducing wasted processing and inspection cost. Sensor importance can be examined from the fitted ExtraTrees model to prioritize investigation, but importance is associational rather than causal; process engineers must validate whether a sensor signal corresponds to a physical defect mechanism.

## 5. Limitations and next steps
The dataset is small relative to its number of sensor features and is strongly imbalanced. Results from one holdout split can vary. Further work should compare models using repeated stratified cross-validation, investigate missingness patterns, tune threshold using out-of-fold predictions, and evaluate calibration. The organizers' hidden-test performance is unknown; no hidden-test score is claimed.
