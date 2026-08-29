
from pathlib import Path
import json

import pandas as pd
import matplotlib.pyplot as plt


# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_FOLDER = PROJECT_ROOT / "results"

POLICY_CONFIG_PATH = (
    RESULTS_FOLDER / "maintenance_policy_config.json"
)

POLICY_RESULTS_PATH = (
    RESULTS_FOLDER / "maintenance_policy_test.csv"
)

RESULTS_FOLDER.mkdir(exist_ok=True)


# --------------------------------------------------
# 2. Machine risk classifier
# --------------------------------------------------

class MachineRiskClassifier:

    def __init__(
        self,
        warning_threshold=0.02,
        critical_threshold=0.10
    ):

        warning_threshold = float(warning_threshold)
        critical_threshold = float(critical_threshold)

        if not 0 <= warning_threshold < critical_threshold <= 1:
            raise ValueError(
                "Thresholds must satisfy: "
                "0 <= warning < critical <= 1"
            )

        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold

    @classmethod
    def from_policy_config(
        cls,
        config_path=POLICY_CONFIG_PATH,
        warning_fraction=0.20
    ):

        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(
                f"Policy configuration not found: {config_path}"
            )

        with open(config_path, "r") as file:
            policy_config = json.load(file)

        critical_threshold = float(
            policy_config["maintenance_threshold"]
        )

        warning_threshold = round(
            critical_threshold * warning_fraction,
            10
        )

        return cls(
            warning_threshold=warning_threshold,
            critical_threshold=critical_threshold
        )

    def classify(self, failure_probability):

        probability = float(failure_probability)

        if not 0 <= probability <= 1:
            raise ValueError(
                "Failure probability must be between 0 and 1."
            )

        if probability >= self.critical_threshold:

            risk_level = "Critical"
            risk_code = 3
            risk_color = "#DC3545"
            recommended_action = "PERFORM MAINTENANCE"
            monitoring_level = "Immediate intervention"
            description = (
                "Failure risk exceeds the economic "
                "maintenance threshold."
            )

        elif probability >= self.warning_threshold:

            risk_level = "Warning"
            risk_code = 2
            risk_color = "#FFA500"
            recommended_action = "MONITOR CLOSELY"
            monitoring_level = "Enhanced monitoring"
            description = (
                "Failure risk is elevated but remains "
                "below the maintenance threshold."
            )

        else:

            risk_level = "Healthy"
            risk_code = 1
            risk_color = "#28A745"
            recommended_action = "CONTINUE OPERATION"
            monitoring_level = "Standard monitoring"
            description = (
                "Machine is operating within the "
                "acceptable risk range."
            )

        return {
            "Risk level": risk_level,
            "Risk code": risk_code,
            "Risk color": risk_color,
            "Risk percentage": probability * 100,
            "Risk recommendation": recommended_action,
            "Monitoring level": monitoring_level,
            "Risk description": description
        }

    def get_configuration(self):

        return {
            "warning_threshold": self.warning_threshold,
            "critical_threshold": self.critical_threshold,
            "healthy_range": (
                f"0 <= p < {self.warning_threshold}"
            ),
            "warning_range": (
                f"{self.warning_threshold} <= p "
                f"< {self.critical_threshold}"
            ),
            "critical_range": (
                f"{self.critical_threshold} <= p <= 1"
            )
        }


# --------------------------------------------------
# 3. Test risk classifier
# --------------------------------------------------

if __name__ == "__main__":

    classifier = MachineRiskClassifier.from_policy_config()

    print("Machine risk classifier")
    print(
        "Warning threshold:",
        round(classifier.warning_threshold, 4)
    )
    print(
        "Critical threshold:",
        round(classifier.critical_threshold, 4)
    )

    print("\nBasic risk-level tests:")

    test_probabilities = [
        0.0037,
        0.0199,
        0.0200,
        0.0500,
        0.0999,
        0.1000,
        0.9500
    ]

    for probability in test_probabilities:

        result = classifier.classify(probability)

        print(
            f"Probability: {probability:.2%} | "
            f"Risk: {result['Risk level']} | "
            f"Action: {result['Risk recommendation']}"
        )

    # --------------------------------------------------
    # 4. Add risk levels to Step 12 results
    # --------------------------------------------------

    if not POLICY_RESULTS_PATH.exists():
        raise FileNotFoundError(
            "Run maintenance_policy.py before Step 13."
        )

    policy_results = pd.read_csv(POLICY_RESULTS_PATH)

    risk_information = policy_results[
        "Failure probability"
    ].apply(
        classifier.classify
    ).apply(
        pd.Series
    )

    risk_results = pd.concat(
        [
            policy_results.reset_index(drop=True),
            risk_information.reset_index(drop=True)
        ],
        axis=1
    )

    risk_results_path = (
        RESULTS_FOLDER / "risk_classification_test.csv"
    )

    risk_results.to_csv(
        risk_results_path,
        index=False
    )

    # --------------------------------------------------
    # 5. Save risk configuration
    # --------------------------------------------------

    risk_config_path = (
        RESULTS_FOLDER / "risk_classification_config.json"
    )

    with open(risk_config_path, "w") as file:
        json.dump(
            classifier.get_configuration(),
            file,
            indent=4
        )

    # --------------------------------------------------
    # 6. Create risk visualization
    # --------------------------------------------------

    color_map = {
        "Healthy": "#28A745",
        "Warning": "#FFA500",
        "Critical": "#DC3545"
    }

    point_colors = risk_results[
        "Risk level"
    ].map(color_map)

    warning_percent = (
        classifier.warning_threshold * 100
    )

    critical_percent = (
        classifier.critical_threshold * 100
    )

    plt.figure(figsize=(11, 6))

    plt.axhspan(
        0,
        warning_percent,
        color="#28A745",
        alpha=0.12,
        label="Healthy"
    )

    plt.axhspan(
        warning_percent,
        critical_percent,
        color="#FFA500",
        alpha=0.15,
        label="Warning"
    )

    plt.axhspan(
        critical_percent,
        100,
        color="#DC3545",
        alpha=0.10,
        label="Critical"
    )

    plt.plot(
        risk_results["Cycle"],
        risk_results["Risk percentage"],
        color="steelblue",
        linewidth=2,
        alpha=0.7
    )

    plt.scatter(
        risk_results["Cycle"],
        risk_results["Risk percentage"],
        c=point_colors,
        s=100,
        edgecolor="black",
        zorder=3
    )

    plt.axhline(
        warning_percent,
        color="#FFA500",
        linestyle="--",
        linewidth=1.5
    )

    plt.axhline(
        critical_percent,
        color="#DC3545",
        linestyle="--",
        linewidth=1.5
    )

    plt.title(
        "Digital Twin Machine Risk Classification"
    )
    plt.xlabel("Simulation cycle")
    plt.ylabel("Predicted failure probability (%)")
    plt.xticks(risk_results["Cycle"])
    plt.ylim(0, 100)
    plt.grid(axis="x", alpha=0.2)
    plt.legend(loc="upper left")
    plt.tight_layout()

    risk_plot_path = (
        RESULTS_FOLDER / "risk_classification_plot.png"
    )

    plt.savefig(
        risk_plot_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # --------------------------------------------------
    # 7. Print results
    # --------------------------------------------------

    print("\nSimulator risk classifications:")

    print(
        risk_results[
            [
                "Cycle",
                "Scenario",
                "Failure probability",
                "Risk level",
                "Risk recommendation"
            ]
        ].to_string(index=False)
    )

    print("\nRisk-level distribution:")

    print(
        risk_results["Risk level"]
        .value_counts()
        .reindex(
            ["Healthy", "Warning", "Critical"],
            fill_value=0
        )
    )

    print("\nRisk results saved at:")
    print(risk_results_path)

    print("\nRisk configuration saved at:")
    print(risk_config_path)

    print("\nRisk visualization saved at:")
    print(risk_plot_path)

    print("\nStep 13 risk classification test completed.")
