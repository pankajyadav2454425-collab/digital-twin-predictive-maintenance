
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.calibration import (
    CalibratedClassifierCV,
    calibration_curve
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
    confusion_matrix,
    classification_report
)


# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_PATH = PROJECT_ROOT / "data" / "ai4i2020.csv"

ORIGINAL_MODEL_PATH = (
    PROJECT_ROOT / "models" / "failure_model.pkl"
)

CALIBRATED_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "failure_model_calibrated.pkl"
)

RESULTS_FOLDER = PROJECT_ROOT / "results"
RESULTS_FOLDER.mkdir(exist_ok=True)


# --------------------------------------------------
# 2. Load and prepare dataset
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)

df["Temperature difference [K]"] = (
    df["Process temperature [K]"]
    - df["Air temperature [K]"]
)

df["Power [W]"] = (
    df["Torque [Nm]"]
    * (
        2
        * np.pi
        * df["Rotational speed [rpm]"]
        / 60
    )
)

df["Overstrain indicator"] = (
    df["Tool wear [min]"]
    * df["Torque [Nm]"]
)

feature_columns = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Temperature difference [K]",
    "Power [W]",
    "Overstrain indicator"
]

X = df[feature_columns].copy()
y = df["Machine failure"].copy()


# --------------------------------------------------
# 3. Recreate the same train-test split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
)

print("Training observations:", len(X_train))
print("Testing observations:", len(X_test))
print("Testing failures:", int(y_test.sum()))


# --------------------------------------------------
# 4. Load original selected model
# --------------------------------------------------

original_model = joblib.load(ORIGINAL_MODEL_PATH)

print("\nOriginal model loaded from:")
print(ORIGINAL_MODEL_PATH)


# --------------------------------------------------
# 5. Original model predictions
# --------------------------------------------------

original_predictions = original_model.predict(X_test)

original_probabilities = (
    original_model.predict_proba(X_test)[:, 1]
)


# --------------------------------------------------
# 6. Calibrate model with five-fold CV
# --------------------------------------------------

print("\nCalibrating model using sigmoid calibration...")

calibrated_model = CalibratedClassifierCV(
    estimator=original_model,
    method="sigmoid",
    cv=5,
    n_jobs=-1
)

calibrated_model.fit(X_train, y_train)

calibrated_predictions = calibrated_model.predict(X_test)

calibrated_probabilities = (
    calibrated_model.predict_proba(X_test)[:, 1]
)

print("Calibration completed successfully.")


# --------------------------------------------------
# 7. Evaluation function
# --------------------------------------------------

def calculate_metrics(
    model_name,
    actual,
    predicted,
    probabilities
):

    return {
        "Model": model_name,
        "Accuracy": accuracy_score(
            actual,
            predicted
        ),
        "Failure Precision": precision_score(
            actual,
            predicted,
            zero_division=0
        ),
        "Failure Recall": recall_score(
            actual,
            predicted,
            zero_division=0
        ),
        "Failure F1": f1_score(
            actual,
            predicted,
            zero_division=0
        ),
        "ROC-AUC": roc_auc_score(
            actual,
            probabilities
        ),
        "PR-AUC": average_precision_score(
            actual,
            probabilities
        ),
        "Brier Score": brier_score_loss(
            actual,
            probabilities
        ),
        "Log Loss": log_loss(
            actual,
            probabilities
        )
    }


# --------------------------------------------------
# 8. Compare original and calibrated models
# --------------------------------------------------

comparison = pd.DataFrame(
    [
        calculate_metrics(
            "Original Random Forest",
            y_test,
            original_predictions,
            original_probabilities
        ),
        calculate_metrics(
            "Calibrated Random Forest",
            y_test,
            calibrated_predictions,
            calibrated_probabilities
        )
    ]
)

comparison.to_csv(
    RESULTS_FOLDER / "calibration_comparison.csv",
    index=False
)

print("\nCalibration comparison:")
print(comparison.round(4).to_string(index=False))


# --------------------------------------------------
# 9. Classification report
# --------------------------------------------------

print("\nCalibrated model classification report:")

print(
    classification_report(
        y_test,
        calibrated_predictions,
        target_names=["Normal", "Failure"],
        zero_division=0
    )
)


# --------------------------------------------------
# 10. Calibration plot
# --------------------------------------------------

original_true_probability, original_mean_probability = (
    calibration_curve(
        y_test,
        original_probabilities,
        n_bins=10,
        strategy="quantile"
    )
)

calibrated_true_probability, calibrated_mean_probability = (
    calibration_curve(
        y_test,
        calibrated_probabilities,
        n_bins=10,
        strategy="quantile"
    )
)

plt.figure(figsize=(8, 7))

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    color="black",
    label="Perfect calibration"
)

plt.plot(
    original_mean_probability,
    original_true_probability,
    marker="o",
    linewidth=2,
    label="Original model"
)

plt.plot(
    calibrated_mean_probability,
    calibrated_true_probability,
    marker="s",
    linewidth=2,
    label="Calibrated model"
)

plt.xlabel("Mean predicted failure probability")
plt.ylabel("Observed failure frequency")
plt.title("Failure Probability Calibration")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    RESULTS_FOLDER
    / "probability_calibration_curve.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# --------------------------------------------------
# 11. Calibrated confusion matrix
# --------------------------------------------------

matrix = confusion_matrix(
    y_test,
    calibrated_predictions
)

plt.figure(figsize=(6, 5))

sns.heatmap(
    matrix,
    annot=True,
    fmt="d",
    cmap="Greens",
    xticklabels=["Normal", "Failure"],
    yticklabels=["Normal", "Failure"]
)

plt.title("Confusion Matrix – Calibrated Model")
plt.xlabel("Predicted class")
plt.ylabel("Actual class")
plt.tight_layout()

plt.savefig(
    RESULTS_FOLDER
    / "calibrated_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# --------------------------------------------------
# 12. Save calibrated model
# --------------------------------------------------

joblib.dump(
    calibrated_model,
    CALIBRATED_MODEL_PATH
)

print("\nCalibrated model saved at:")
print(CALIBRATED_MODEL_PATH)


# --------------------------------------------------
# 13. Save calibrated test predictions
# --------------------------------------------------

prediction_results = X_test.copy()

prediction_results["Actual failure"] = y_test.values

prediction_results["Original probability"] = (
    original_probabilities
)

prediction_results["Calibrated probability"] = (
    calibrated_probabilities
)

prediction_results["Calibrated prediction"] = (
    calibrated_predictions
)

prediction_results.to_csv(
    RESULTS_FOLDER
    / "calibrated_test_predictions.csv",
    index=False
)


# --------------------------------------------------
# 14. Save calibration metadata
# --------------------------------------------------

metadata = {
    "calibration_method": "sigmoid",
    "cross_validation_folds": 5,
    "calibrated_model": str(
        CALIBRATED_MODEL_PATH
    ),
    "testing_observations": int(len(X_test)),
    "testing_failures": int(y_test.sum())
}

with open(
    RESULTS_FOLDER / "calibration_metadata.json",
    "w"
) as file:
    json.dump(metadata, file, indent=4)


print("\nStep 8 completed successfully.")
