
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)


# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "ai4i2020.csv"
MODEL_FOLDER = PROJECT_ROOT / "models"
RESULTS_FOLDER = PROJECT_ROOT / "results"

MODEL_FOLDER.mkdir(exist_ok=True)
RESULTS_FOLDER.mkdir(exist_ok=True)


# --------------------------------------------------
# 2. Load dataset
# --------------------------------------------------

print("Loading dataset from:", DATA_PATH)

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)
print("Machine failures:", int(df["Machine failure"].sum()))
print(
    "Failure percentage:",
    round(df["Machine failure"].mean() * 100, 2),
    "%"
)


# --------------------------------------------------
# 3. Feature engineering
# --------------------------------------------------

df["Temperature difference [K]"] = (
    df["Process temperature [K]"]
    - df["Air temperature [K]"]
)

df["Power [W]"] = (
    df["Torque [Nm]"]
    * (2 * np.pi * df["Rotational speed [rpm]"] / 60)
)

df["Overstrain indicator"] = (
    df["Tool wear [min]"]
    * df["Torque [Nm]"]
)


# --------------------------------------------------
# 4. Select features and target
# --------------------------------------------------

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

target_column = "Machine failure"

X = df[feature_columns].copy()
y = df[target_column].copy()

categorical_features = ["Type"]

numerical_features = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Temperature difference [K]",
    "Power [W]",
    "Overstrain indicator"
]


# --------------------------------------------------
# 5. Train-test split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
)

print("\nTraining observations:", len(X_train))
print("Testing observations:", len(X_test))
print("Testing failures:", int(y_test.sum()))


# --------------------------------------------------
# 6. Preprocessing functions
# --------------------------------------------------

def create_preprocessor(scale_numeric=False):

    numeric_transformer = (
        StandardScaler()
        if scale_numeric
        else "passthrough"
    )

    return ColumnTransformer(
        transformers=[
            (
                "type",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                ),
                categorical_features
            ),
            (
                "numeric",
                numeric_transformer,
                numerical_features
            )
        ],
        remainder="drop",
        verbose_feature_names_out=False
    )


# --------------------------------------------------
# 7. Create models
# --------------------------------------------------

logistic_model = Pipeline(
    steps=[
        (
            "preprocessor",
            create_preprocessor(scale_numeric=True)
        ),
        (
            "classifier",
            LogisticRegression(
                class_weight="balanced",
                max_iter=2000,
                random_state=42
            )
        )
    ]
)

random_forest_model = Pipeline(
    steps=[
        (
            "preprocessor",
            create_preprocessor(scale_numeric=False)
        ),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=300,
                max_depth=12,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)

models = {
    "Logistic Regression": logistic_model,
    "Random Forest": random_forest_model
}


# --------------------------------------------------
# 8. Train and evaluate models
# --------------------------------------------------

model_results = []
trained_models = {}

best_model = None
best_model_name = None
best_f1 = -1

for model_name, model in models.items():

    print("\n" + "=" * 60)
    print("Training:", model_name)

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )
    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )
    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )
    roc_auc = roc_auc_score(y_test, probabilities)
    pr_auc = average_precision_score(
        y_test,
        probabilities
    )

    model_results.append(
        {
            "Model": model_name,
            "Accuracy": accuracy,
            "Failure Precision": precision,
            "Failure Recall": recall,
            "Failure F1": f1,
            "ROC-AUC": roc_auc,
            "PR-AUC": pr_auc
        }
    )

    trained_models[model_name] = model

    print(
        classification_report(
            y_test,
            predictions,
            target_names=["Normal", "Failure"],
            zero_division=0
        )
    )

    # Save confusion matrix
    matrix = confusion_matrix(y_test, predictions)

    plt.figure(figsize=(6, 5))

    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Normal", "Failure"],
        yticklabels=["Normal", "Failure"]
    )

    plt.title(f"Confusion Matrix – {model_name}")
    plt.xlabel("Predicted class")
    plt.ylabel("Actual class")
    plt.tight_layout()

    filename = (
        model_name.lower()
        .replace(" ", "_")
        + "_confusion_matrix.png"
    )

    plt.savefig(
        RESULTS_FOLDER / filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # Select best model using failure-class F1
    if f1 > best_f1:
        best_f1 = f1
        best_model = model
        best_model_name = model_name


# --------------------------------------------------
# 9. Save model comparison
# --------------------------------------------------

results_df = pd.DataFrame(model_results)

results_df = results_df.sort_values(
    by="Failure F1",
    ascending=False
)

results_df.to_csv(
    RESULTS_FOLDER / "model_comparison.csv",
    index=False
)

print("\nModel comparison:")
print(results_df.round(4).to_string(index=False))


# --------------------------------------------------
# 10. Save best model
# --------------------------------------------------

MODEL_PATH = MODEL_FOLDER / "failure_model.pkl"

joblib.dump(best_model, MODEL_PATH)

print("\nBest model:", best_model_name)
print("Best failure F1:", round(best_f1, 4))
print("Model saved at:", MODEL_PATH)


# --------------------------------------------------
# 11. Save test predictions
# --------------------------------------------------

best_predictions = best_model.predict(X_test)
best_probabilities = best_model.predict_proba(X_test)[:, 1]

test_results = X_test.copy()
test_results["Actual failure"] = y_test.values
test_results["Predicted failure"] = best_predictions
test_results["Failure probability"] = best_probabilities

test_results.to_csv(
    RESULTS_FOLDER / "test_predictions.csv",
    index=False
)


# --------------------------------------------------
# 12. Save Random Forest feature importance
# --------------------------------------------------

rf_pipeline = trained_models["Random Forest"]

rf_preprocessor = rf_pipeline.named_steps["preprocessor"]
rf_classifier = rf_pipeline.named_steps["classifier"]

feature_names = (
    rf_preprocessor.get_feature_names_out()
)

feature_importance = pd.DataFrame(
    {
        "Feature": feature_names,
        "Importance": rf_classifier.feature_importances_
    }
).sort_values(
    by="Importance",
    ascending=False
)

feature_importance.to_csv(
    RESULTS_FOLDER / "feature_importance.csv",
    index=False
)

plt.figure(figsize=(10, 6))

sns.barplot(
    data=feature_importance,
    x="Importance",
    y="Feature",
    color="steelblue"
)

plt.title("Random Forest Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()

plt.savefig(
    RESULTS_FOLDER / "feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# --------------------------------------------------
# 13. Save model information
# --------------------------------------------------

metadata = {
    "selected_model": best_model_name,
    "selection_metric": "Failure-class F1-score",
    "best_failure_f1": float(best_f1),
    "training_observations": int(len(X_train)),
    "testing_observations": int(len(X_test)),
    "features": feature_columns,
    "target": target_column
}

with open(
    RESULTS_FOLDER / "model_metadata.json",
    "w"
) as file:
    json.dump(metadata, file, indent=4)


print("\nAll outputs saved successfully.")
print("Results folder:", RESULTS_FOLDER)
