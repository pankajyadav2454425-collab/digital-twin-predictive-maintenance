
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split


# --------------------------------------------------
# 1. Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_PATH = (
    PROJECT_ROOT / "data" / "ai4i2020.csv"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "failure_model_calibrated.pkl"
)

RESULTS_FOLDER = PROJECT_ROOT / "results"

POLICY_CONFIG_PATH = (
    RESULTS_FOLDER / "maintenance_policy_config.json"
)

RESULTS_FOLDER.mkdir(exist_ok=True)


# --------------------------------------------------
# 2. Load maintenance costs
# --------------------------------------------------

with open(POLICY_CONFIG_PATH, "r") as file:
    policy_config = json.load(file)

MAINTENANCE_COST = float(
    policy_config["preventive_maintenance_cost"]
)

FAILURE_COST = float(
    policy_config["failure_cost"]
)

PREDICTIVE_THRESHOLD = float(
    policy_config["maintenance_threshold"]
)

PREVENTIVE_WEAR_THRESHOLD = 200.0


# --------------------------------------------------
# 3. Load and prepare AI4I data
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

_, X_test, _, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
)

print("Evaluation observations:", len(X_test))
print("Actual machine failures:", int(y_test.sum()))


# --------------------------------------------------
# 4. Generate calibrated failure probabilities
# --------------------------------------------------

model = joblib.load(MODEL_PATH)

failure_probabilities = model.predict_proba(
    X_test
)[:, 1]

evaluation_df = X_test.copy()

evaluation_df["Actual failure"] = (
    y_test.astype(int)
)

evaluation_df["Failure probability"] = (
    failure_probabilities
)

evaluation_df = evaluation_df.reset_index().rename(
    columns={
        "index": "Dataset index"
    }
)


# --------------------------------------------------
# 5. Maintenance-strategy decisions
# --------------------------------------------------

actual_failure = (
    evaluation_df["Actual failure"] == 1
)

# Corrective: no planned intervention
corrective_action = pd.Series(
    False,
    index=evaluation_df.index
)

# Preventive: fixed tool-wear rule
preventive_action = (
    evaluation_df["Tool wear [min]"]
    >= PREVENTIVE_WEAR_THRESHOLD
)

# Predictive: calibrated economic-risk rule
predictive_action = (
    evaluation_df["Failure probability"]
    >= PREDICTIVE_THRESHOLD
)

evaluation_df["Corrective action"] = np.where(
    actual_failure,
    "REPAIR AFTER FAILURE",
    "CONTINUE OPERATION"
)

evaluation_df["Preventive action"] = np.where(
    preventive_action,
    "PERFORM MAINTENANCE",
    "CONTINUE OPERATION"
)

evaluation_df["Predictive action"] = np.where(
    predictive_action,
    "PERFORM MAINTENANCE",
    "CONTINUE OPERATION"
)


# --------------------------------------------------
# 6. Strategy-evaluation function
# --------------------------------------------------

def evaluate_strategy(
    strategy_name,
    planned_action
):

    planned_action = planned_action.astype(bool)

    total_observations = len(evaluation_df)
    total_failures = int(actual_failure.sum())

    planned_maintenance = int(
        planned_action.sum()
    )

    prevented_failures = int(
        (
            planned_action
            & actual_failure
        ).sum()
    )

    unplanned_failures = int(
        (
            ~planned_action
            & actual_failure
        ).sum()
    )

    unnecessary_maintenance = int(
        (
            planned_action
            & ~actual_failure
        ).sum()
    )

    corrective_repairs = unplanned_failures

    total_interventions = (
        planned_maintenance
        + corrective_repairs
    )

    maintenance_cost_total = (
        planned_maintenance
        * MAINTENANCE_COST
    )

    failure_cost_total = (
        unplanned_failures
        * FAILURE_COST
    )

    total_cost = (
        maintenance_cost_total
        + failure_cost_total
    )

    failure_coverage = (
        prevented_failures / total_failures
        if total_failures > 0
        else 0
    )

    maintenance_precision = (
        prevented_failures / planned_maintenance
        if planned_maintenance > 0
        else 0
    )

    return {
        "Strategy": strategy_name,
        "Evaluated cycles": total_observations,
        "Actual failures": total_failures,
        "Planned maintenance": planned_maintenance,
        "Prevented failures": prevented_failures,
        "Unplanned failures": unplanned_failures,
        "Corrective repairs": corrective_repairs,
        "Unnecessary maintenance": (
            unnecessary_maintenance
        ),
        "Total interventions": total_interventions,
        "Failure coverage": failure_coverage,
        "Maintenance precision": (
            maintenance_precision
        ),
        "Maintenance cost": (
            maintenance_cost_total
        ),
        "Failure cost": failure_cost_total,
        "Total cost": total_cost,
        "Cost per cycle": (
            total_cost / total_observations
        )
    }


# --------------------------------------------------
# 7. Compare strategies
# --------------------------------------------------

strategy_results = [
    evaluate_strategy(
        "Corrective",
        corrective_action
    ),
    evaluate_strategy(
        "Preventive (tool-wear rule)",
        preventive_action
    ),
    evaluate_strategy(
        "Predictive digital twin",
        predictive_action
    )
]

comparison_df = pd.DataFrame(
    strategy_results
)

corrective_total_cost = float(
    comparison_df.loc[
        comparison_df["Strategy"] == "Corrective",
        "Total cost"
    ].iloc[0]
)

comparison_df["Savings vs corrective"] = (
    corrective_total_cost
    - comparison_df["Total cost"]
)

comparison_df["Savings percentage"] = (
    comparison_df["Savings vs corrective"]
    / corrective_total_cost
)

comparison_df = comparison_df.sort_values(
    "Total cost"
).reset_index(drop=True)

best_strategy = comparison_df.iloc[0]


# --------------------------------------------------
# 8. Validate calculations
# --------------------------------------------------

for _, row in comparison_df.iterrows():

    assert (
        row["Prevented failures"]
        + row["Unplanned failures"]
        == row["Actual failures"]
    )

    assert row["Total cost"] == (
        row["Maintenance cost"]
        + row["Failure cost"]
    )

print("\nMaintenance-strategy comparison:")

print(
    comparison_df[
        [
            "Strategy",
            "Planned maintenance",
            "Prevented failures",
            "Unplanned failures",
            "Unnecessary maintenance",
            "Failure coverage",
            "Total cost",
            "Savings vs corrective"
        ]
    ].round(4).to_string(index=False)
)

print("\nBest economic strategy:")
print(best_strategy["Strategy"])

print(
    "Minimum total cost:",
    f"₹{best_strategy['Total cost']:,.0f}"
)

print(
    "Savings compared with corrective:",
    f"₹{best_strategy['Savings vs corrective']:,.0f}"
)


# --------------------------------------------------
# 9. Save results
# --------------------------------------------------

comparison_path = (
    RESULTS_FOLDER
    / "maintenance_strategy_comparison.csv"
)

details_path = (
    RESULTS_FOLDER
    / "maintenance_strategy_decisions.csv"
)

config_path = (
    RESULTS_FOLDER
    / "maintenance_strategy_config.json"
)

comparison_df.to_csv(
    comparison_path,
    index=False
)

evaluation_df.to_csv(
    details_path,
    index=False
)

strategy_config = {
    "maintenance_cost": MAINTENANCE_COST,
    "failure_cost": FAILURE_COST,
    "predictive_threshold": PREDICTIVE_THRESHOLD,
    "preventive_tool_wear_threshold": (
        PREVENTIVE_WEAR_THRESHOLD
    ),
    "evaluation_observations": int(
        len(evaluation_df)
    ),
    "actual_failures": int(
        actual_failure.sum()
    ),
    "best_strategy": best_strategy["Strategy"],
    "best_total_cost": float(
        best_strategy["Total cost"]
    ),
    "comparison_assumption": (
        "A planned intervention on a failure-labelled "
        "observation prevents that failure."
    )
}

with open(config_path, "w") as file:
    json.dump(
        strategy_config,
        file,
        indent=4
    )


# --------------------------------------------------
# 10. Total-cost visualization
# --------------------------------------------------

cost_plot_data = comparison_df.copy()

cost_plot_data["Total cost (₹ million)"] = (
    cost_plot_data["Total cost"] / 1_000_000
)

plt.figure(figsize=(10, 6))

ax = sns.barplot(
    data=cost_plot_data,
    x="Strategy",
    y="Total cost (₹ million)",
    hue="Strategy",
    palette=[
        "#28A745",
        "#FFA500",
        "#DC3545"
    ],
    legend=False
)

for container in ax.containers:
    ax.bar_label(
        container,
        fmt="%.2f",
        padding=3
    )

plt.title(
    "Total Cost of Maintenance Strategies"
)
plt.xlabel("Maintenance strategy")
plt.ylabel("Total cost (₹ million)")
plt.xticks(rotation=10)
plt.tight_layout()

cost_plot_path = (
    RESULTS_FOLDER
    / "maintenance_strategy_cost.png"
)

plt.savefig(
    cost_plot_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# --------------------------------------------------
# 11. Outcome visualization
# --------------------------------------------------

outcome_columns = [
    "Planned maintenance",
    "Prevented failures",
    "Unplanned failures",
    "Unnecessary maintenance"
]

outcome_data = comparison_df[
    ["Strategy"] + outcome_columns
].melt(
    id_vars="Strategy",
    var_name="Outcome",
    value_name="Count"
)

plt.figure(figsize=(12, 7))

sns.barplot(
    data=outcome_data,
    x="Strategy",
    y="Count",
    hue="Outcome"
)

plt.title(
    "Maintenance Interventions and Failure Outcomes"
)
plt.xlabel("Maintenance strategy")
plt.ylabel("Number of observations")
plt.xticks(rotation=10)
plt.legend(title="Outcome")
plt.tight_layout()

outcome_plot_path = (
    RESULTS_FOLDER
    / "maintenance_strategy_outcomes.png"
)

plt.savefig(
    outcome_plot_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print("\nComparison saved at:")
print(comparison_path)

print("\nDetailed decisions saved at:")
print(details_path)

print("\nConfiguration saved at:")
print(config_path)

print("\nPlots saved at:")
print(cost_plot_path)
print(outcome_plot_path)

print("\nStep 16 strategy comparison completed.")
