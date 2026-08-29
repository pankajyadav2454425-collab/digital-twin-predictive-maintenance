
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "failure_model_calibrated.pkl"
)

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "ai4i2020.csv"
)

RESULTS_FOLDER = PROJECT_ROOT / "results"
RESULTS_FOLDER.mkdir(exist_ok=True)


class LocalPredictionExplainer:

    def __init__(
        self,
        model_path=MODEL_PATH,
        data_path=DATA_PATH
    ):

        self.model = joblib.load(model_path)
        self.dataset = pd.read_csv(data_path)

        self.raw_features = [
            "Type",
            "Air temperature [K]",
            "Process temperature [K]",
            "Rotational speed [rpm]",
            "Torque [Nm]",
            "Tool wear [min]"
        ]

        self.reference_state = {
            "Type": self.dataset["Type"].mode().iloc[0],
            "Air temperature [K]": float(
                self.dataset[
                    "Air temperature [K]"
                ].median()
            ),
            "Process temperature [K]": float(
                self.dataset[
                    "Process temperature [K]"
                ].median()
            ),
            "Rotational speed [rpm]": float(
                self.dataset[
                    "Rotational speed [rpm]"
                ].median()
            ),
            "Torque [Nm]": float(
                self.dataset["Torque [Nm]"].median()
            ),
            "Tool wear [min]": float(
                self.dataset[
                    "Tool wear [min]"
                ].median()
            )
        }

    def build_model_input(self, raw_state):

        state = raw_state.copy()

        temperature_difference = (
            state["Process temperature [K]"]
            - state["Air temperature [K]"]
        )

        power = (
            state["Torque [Nm]"]
            * (
                2
                * np.pi
                * state["Rotational speed [rpm]"]
                / 60
            )
        )

        overstrain_indicator = (
            state["Tool wear [min]"]
            * state["Torque [Nm]"]
        )

        return pd.DataFrame(
            {
                "Type": [state["Type"]],
                "Air temperature [K]": [
                    state["Air temperature [K]"]
                ],
                "Process temperature [K]": [
                    state["Process temperature [K]"]
                ],
                "Rotational speed [rpm]": [
                    state["Rotational speed [rpm]"]
                ],
                "Torque [Nm]": [
                    state["Torque [Nm]"]
                ],
                "Tool wear [min]": [
                    state["Tool wear [min]"]
                ],
                "Temperature difference [K]": [
                    temperature_difference
                ],
                "Power [W]": [power],
                "Overstrain indicator": [
                    overstrain_indicator
                ]
            }
        )

    def predict_probability(self, raw_state):

        model_input = self.build_model_input(raw_state)

        return float(
            self.model.predict_proba(
                model_input
            )[0, 1]
        )

    def explain(self, raw_state):

        missing_features = [
            feature
            for feature in self.raw_features
            if feature not in raw_state
        ]

        if missing_features:
            raise ValueError(
                f"Missing features: {missing_features}"
            )

        original_probability = (
            self.predict_probability(raw_state)
        )

        explanation_records = []

        for feature in self.raw_features:

            counterfactual_state = raw_state.copy()

            reference_value = (
                self.reference_state[feature]
            )

            counterfactual_state[feature] = (
                reference_value
            )

            counterfactual_probability = (
                self.predict_probability(
                    counterfactual_state
                )
            )

            contribution = (
                original_probability
                - counterfactual_probability
            )

            if contribution > 0.0001:
                direction = "Increases failure risk"
                color = "#DC3545"

            elif contribution < -0.0001:
                direction = "Reduces failure risk"
                color = "#28A745"

            else:
                direction = "Little local effect"
                color = "#6C757D"

            explanation_records.append(
                {
                    "Feature": feature,
                    "Current value": raw_state[feature],
                    "Reference value": reference_value,
                    "Original probability": (
                        original_probability
                    ),
                    "Probability with reference": (
                        counterfactual_probability
                    ),
                    "Contribution": contribution,
                    "Contribution percentage points": (
                        contribution * 100
                    ),
                    "Absolute contribution": abs(
                        contribution
                    ),
                    "Direction": direction,
                    "Color": color
                }
            )

        explanation_df = pd.DataFrame(
            explanation_records
        )

        explanation_df = explanation_df.sort_values(
            "Absolute contribution",
            ascending=False
        ).reset_index(drop=True)

        return explanation_df

    def create_plot(
        self,
        explanation_df,
        title,
        output_path
    ):

        plot_data = explanation_df.sort_values(
            "Contribution percentage points"
        )

        colors = plot_data["Color"].tolist()

        plt.figure(figsize=(10, 6))

        plt.barh(
            plot_data["Feature"],
            plot_data[
                "Contribution percentage points"
            ],
            color=colors
        )

        plt.axvline(
            0,
            color="black",
            linewidth=1
        )

        plt.title(title)
        plt.xlabel(
            "Change in failure probability "
            "(percentage points)"
        )
        plt.ylabel("Machine feature")
        plt.grid(axis="x", alpha=0.25)
        plt.tight_layout()

        plt.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()


if __name__ == "__main__":

    explainer = LocalPredictionExplainer()

    test_states = {
        "Healthy": {
            "Type": "M",
            "Air temperature [K]": 298.5,
            "Process temperature [K]": 309.1,
            "Rotational speed [rpm]": 1705,
            "Torque [Nm]": 33.66,
            "Tool wear [min]": 8.0
        },
        "Critical": {
            "Type": "L",
            "Air temperature [K]": 299.45,
            "Process temperature [K]": 309.10,
            "Rotational speed [rpm]": 1259,
            "Torque [Nm]": 68.46,
            "Tool wear [min]": 27.0
        }
    }

    print("Reference machine state:")

    for feature, value in (
        explainer.reference_state.items()
    ):
        print(f"{feature}: {value}")

    for state_name, state in test_states.items():

        probability = (
            explainer.predict_probability(state)
        )

        explanation = explainer.explain(state)

        csv_path = (
            RESULTS_FOLDER
            / f"{state_name.lower()}_local_explanation.csv"
        )

        plot_path = (
            RESULTS_FOLDER
            / f"{state_name.lower()}_local_explanation.png"
        )

        explanation.to_csv(
            csv_path,
            index=False
        )

        explainer.create_plot(
            explanation_df=explanation,
            title=(
                f"{state_name} State – "
                "Local Failure-Risk Explanation"
            ),
            output_path=plot_path
        )

        print("\n" + "=" * 65)
        print(
            f"{state_name} failure probability: "
            f"{probability:.4f}"
        )

        print(
            explanation[
                [
                    "Feature",
                    "Contribution percentage points",
                    "Direction"
                ]
            ].to_string(index=False)
        )

        print("\nCSV saved at:", csv_path)
        print("Plot saved at:", plot_path)

    print(
        "\nStep 15 local explanation test completed."
    )
