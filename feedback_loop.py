
from pathlib import Path
import json

import pandas as pd
import matplotlib.pyplot as plt

from maintenance_policy import CostAwareMaintenancePolicy
from risk_classifier import MachineRiskClassifier
from explainability import LocalPredictionExplainer


# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_FOLDER = PROJECT_ROOT / "results"

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

POLICY_CONFIG_PATH = (
    RESULTS_FOLDER / "maintenance_policy_config.json"
)

RESULTS_FOLDER.mkdir(exist_ok=True)


# --------------------------------------------------
# 2. Closed-loop digital twin
# --------------------------------------------------

class DigitalTwinFeedbackLoop:

    def __init__(self):

        with open(
            POLICY_CONFIG_PATH,
            "r"
        ) as file:
            policy_config = json.load(file)

        self.maintenance_cost = float(
            policy_config[
                "preventive_maintenance_cost"
            ]
        )

        self.failure_cost = float(
            policy_config["failure_cost"]
        )

        self.explainer = LocalPredictionExplainer(
            model_path=MODEL_PATH,
            data_path=DATA_PATH
        )

        self.policy = CostAwareMaintenancePolicy(
            preventive_maintenance_cost=(
                self.maintenance_cost
            ),
            failure_cost=self.failure_cost,
            currency="INR"
        )

        self.critical_threshold = round(
            self.policy.maintenance_threshold,
            10
        )

        self.warning_threshold = round(
            self.critical_threshold * 0.20,
            10
        )

        self.risk_classifier = (
            MachineRiskClassifier(
                warning_threshold=(
                    self.warning_threshold
                ),
                critical_threshold=(
                    self.critical_threshold
                )
            )
        )

        self.history = []
        self.maintenance_events = []

        self.sequence_number = 0
        self.maintenance_count = 0
        self.cumulative_realized_cost = 0.0

        self.healthy_states = [
            {
                "Type": "L",
                "Air temperature [K]": 296.75,
                "Process temperature [K]": 307.65,
                "Rotational speed [rpm]": 1885,
                "Torque [Nm]": 22.02,
                "Tool wear [min]": 2.0
            },
            {
                "Type": "M",
                "Air temperature [K]": 298.45,
                "Process temperature [K]": 308.88,
                "Rotational speed [rpm]": 1769,
                "Torque [Nm]": 28.37,
                "Tool wear [min]": 5.0
            },
            {
                "Type": "M",
                "Air temperature [K]": 298.55,
                "Process temperature [K]": 309.14,
                "Rotational speed [rpm]": 1705,
                "Torque [Nm]": 33.66,
                "Tool wear [min]": 8.0
            },
            {
                "Type": "L",
                "Air temperature [K]": 299.48,
                "Process temperature [K]": 309.89,
                "Rotational speed [rpm]": 1688,
                "Torque [Nm]": 33.74,
                "Tool wear [min]": 10.0
            },
            {
                "Type": "H",
                "Air temperature [K]": 300.20,
                "Process temperature [K]": 309.92,
                "Rotational speed [rpm]": 1658,
                "Torque [Nm]": 33.14,
                "Tool wear [min]": 15.0
            }
        ]

        self.critical_state = {
            "Type": "L",
            "Air temperature [K]": 299.45,
            "Process temperature [K]": 309.10,
            "Rotational speed [rpm]": 1259,
            "Torque [Nm]": 68.46,
            "Tool wear [min]": 27.0
        }

        self.post_maintenance_state = {
            "Type": "L",
            "Air temperature [K]": 298.70,
            "Process temperature [K]": 310.10,
            "Rotational speed [rpm]": 1401,
            "Torque [Nm]": 45.30,
            "Tool wear [min]": 0.0
        }

    def generate_state(self, cycle):

        if self.maintenance_count == 0:

            if cycle <= len(self.healthy_states):
                return self.healthy_states[
                    cycle - 1
                ].copy()

            return self.critical_state.copy()

        recovery_step = cycle - 6

        recovered_state = (
            self.post_maintenance_state.copy()
        )

        recovered_state["Tool wear [min]"] = (
            max(0.0, recovery_step * 2.0)
        )

        recovered_state[
            "Rotational speed [rpm]"
        ] = (
            self.post_maintenance_state[
                "Rotational speed [rpm]"
            ]
            + recovery_step * 15
        )

        recovered_state["Torque [Nm]"] = max(
            38.0,
            self.post_maintenance_state[
                "Torque [Nm]"
            ]
            - recovery_step
        )

        return recovered_state

    def get_top_risk_driver(self, state):

        explanation = self.explainer.explain(
            state
        )

        positive_drivers = explanation[
            explanation[
                "Contribution percentage points"
            ] > 0
        ]

        if positive_drivers.empty:
            return "No positive local driver"

        return positive_drivers.iloc[0][
            "Feature"
        ]

    def add_history_row(
        self,
        cycle,
        event,
        state,
        probability,
        risk_result,
        policy_result,
        top_driver="Not evaluated",
        realized_cost=0.0
    ):

        self.sequence_number += 1

        self.history.append(
            {
                "Sequence": self.sequence_number,
                "Cycle": cycle,
                "Event": event,
                "Type": state["Type"],
                "Air temperature [K]": state[
                    "Air temperature [K]"
                ],
                "Process temperature [K]": state[
                    "Process temperature [K]"
                ],
                "Rotational speed [rpm]": state[
                    "Rotational speed [rpm]"
                ],
                "Torque [Nm]": state[
                    "Torque [Nm]"
                ],
                "Tool wear [min]": state[
                    "Tool wear [min]"
                ],
                "Failure probability": probability,
                "Risk level": risk_result[
                    "Risk level"
                ],
                "Recommended action": policy_result[
                    "Recommended action"
                ],
                "Expected continue cost": (
                    policy_result[
                        "Expected continue cost"
                    ]
                ),
                "Expected selected cost": (
                    policy_result[
                        "Selected expected cost"
                    ]
                ),
                "Expected savings": policy_result[
                    "Expected savings"
                ],
                "Top risk driver": top_driver,
                "Realized event cost": realized_cost,
                "Cumulative realized cost": (
                    self.cumulative_realized_cost
                ),
                "Maintenance count": (
                    self.maintenance_count
                )
            }
        )

    def apply_maintenance(
        self,
        cycle,
        before_probability,
        before_risk,
        expected_savings
    ):

        maintained_state = (
            self.post_maintenance_state.copy()
        )

        after_probability = (
            self.explainer.predict_probability(
                maintained_state

            )
        )

        after_risk = (
            self.risk_classifier.classify(
                after_probability
            )
        )

        after_policy = self.policy.evaluate(
            after_probability
        )

        self.maintenance_count += 1
        self.cumulative_realized_cost += (
            self.maintenance_cost
        )

        self.add_history_row(
            cycle=cycle,
            event="Maintenance",
            state=maintained_state,
            probability=after_probability,
            risk_result=after_risk,
            policy_result=after_policy,
            top_driver="Maintenance reset",
            realized_cost=self.maintenance_cost
        )

        risk_reduction = (
            before_probability
            - after_probability
        )

        maintenance_event = {
            "Cycle": cycle,
            "Probability before": (
                before_probability
            ),
            "Probability after": (
                after_probability
            ),
            "Risk before": before_risk,
            "Risk after": after_risk[
                "Risk level"
            ],
            "Risk reduction": risk_reduction,
            "Risk reduction percentage points": (
                risk_reduction * 100
            ),
            "Tool wear after": maintained_state[
                "Tool wear [min]"
            ],
            "Expected savings": expected_savings
        }

        self.maintenance_events.append(
            maintenance_event
        )

        return maintained_state

    def run(self, total_cycles=10):

        print("Closed-loop digital twin started.")
        print(
            "Warning threshold:",
            self.warning_threshold
        )
        print(
            "Maintenance threshold:",
            self.critical_threshold
        )

        for cycle in range(
            1,
            total_cycles + 1
        ):

            state = self.generate_state(cycle)

            probability = (
                self.explainer.predict_probability(
                    state
                )
            )

            risk_result = (
                self.risk_classifier.classify(
                    probability
                )
            )

            policy_result = self.policy.evaluate(
                probability
            )

            if risk_result["Risk level"] in [
                "Warning",
                "Critical"
            ]:
                top_driver = (
                    self.get_top_risk_driver(
                        state
                    )
                )
            else:
                top_driver = "Not required"

            self.add_history_row(
                cycle=cycle,
                event="Operation",
                state=state,
                probability=probability,
                risk_result=risk_result,
                policy_result=policy_result,
                top_driver=top_driver
            )

            print(
                f"Cycle {cycle:02d} | "
                f"Wear: {state['Tool wear [min]']:.1f} | "
                f"Speed: "
                f"{state['Rotational speed [rpm]']} | "
                f"Torque: "
                f"{state['Torque [Nm]']:.2f} | "
                f"Probability: {probability:.4f} | "
                f"Risk: {risk_result['Risk level']} | "
                f"Action: "
                f"{policy_result['Recommended action']}"
            )

            if policy_result[
                "Perform maintenance"
            ]:

                print(
                    "  Automatic maintenance triggered."
                )

                self.apply_maintenance(
                    cycle=cycle,
                    before_probability=probability,
                    before_risk=risk_result[
                        "Risk level"
                    ],
                    expected_savings=policy_result[
                        "Expected savings"
                    ]
                )

                maintenance_result = (
                    self.maintenance_events[-1]
                )

                print(
                    "  Probability after maintenance:",
                    round(
                        maintenance_result[
                            "Probability after"
                        ],
                        4
                    )
                )

                print(
                    "  Risk after maintenance:",
                    maintenance_result[
                        "Risk after"
                    ]
                )

        return pd.DataFrame(self.history)


# --------------------------------------------------
# 3. Run closed-loop test
# --------------------------------------------------

if __name__ == "__main__":

    digital_twin = DigitalTwinFeedbackLoop()

    history_df = digital_twin.run(
        total_cycles=10
    )

    history_path = (
        RESULTS_FOLDER
        / "feedback_loop_history.csv"
    )

    history_df.to_csv(
        history_path,
        index=False
    )

    maintenance_events = (
        digital_twin.maintenance_events
    )

    if not maintenance_events:
        raise AssertionError(
            "No maintenance event was triggered."
        )

    first_event = maintenance_events[0]

    operation_rows = history_df[
        history_df["Event"] == "Operation"
    ]

    final_operation = operation_rows.iloc[-1]

    validation_checks = {
        "maintenance_triggered": (
            digital_twin.maintenance_count >= 1
        ),
        "critical_risk_detected": (
            first_event["Risk before"]
            == "Critical"
        ),
        "tool_wear_reset": (
            first_event["Tool wear after"]
            == 0.0
        ),
        "risk_reduced": (
            first_event["Probability after"]
            < first_event["Probability before"]
        ),
        "returned_below_warning_threshold": (
            first_event["Probability after"]
            < digital_twin.warning_threshold
        ),
        "returned_to_healthy_state": (
            first_event["Risk after"]
            == "Healthy"
        ),
        "final_state_healthy": (
            final_operation["Risk level"]
            == "Healthy"
        )
    }

    feedback_loop_passed = all(
        validation_checks.values()
    )

    summary = {
        "feedback_loop_passed": (
            feedback_loop_passed
        ),
        "operating_cycles": int(
            len(operation_rows)
        ),
        "maintenance_events": int(
            digital_twin.maintenance_count
        ),
        "maintenance_threshold": (
            digital_twin.critical_threshold
        ),
        "probability_before_maintenance": (
            float(
                first_event[
                    "Probability before"
                ]
            )
        ),
        "probability_after_maintenance": (
            float(
                first_event[
                    "Probability after"
                ]
            )
        ),
        "risk_reduction_percentage_points": (
            float(
                first_event[
                    "Risk reduction percentage points"
                ]
            )
        ),
        "risk_before": first_event[
            "Risk before"
        ],
        "risk_after": first_event[
            "Risk after"
        ],
        "expected_savings_at_intervention": (
            float(
                first_event[
                    "Expected savings"
                ]
            )
        ),
        "total_realized_maintenance_cost": (
            float(
                digital_twin.cumulative_realized_cost
            )
        ),
        "final_failure_probability": float(
            final_operation[
                "Failure probability"
            ]
        ),
        "final_risk_level": final_operation[
            "Risk level"
        ],
        "validation_checks": validation_checks
    }

    summary_path = (
        RESULTS_FOLDER
        / "feedback_loop_summary.json"
    )

    with open(summary_path, "w") as file:
        json.dump(
            summary,
            file,
            indent=4
        )

    # --------------------------------------------------
    # 4. Failure-probability plot
    # --------------------------------------------------

    plt.figure(figsize=(12, 6))

    plt.plot(
        history_df["Sequence"],
        history_df[
            "Failure probability"
        ] * 100,
        marker="o",
        linewidth=2.5,
        color="steelblue"
    )

    plt.axhline(
        digital_twin.warning_threshold * 100,
        color="#FFA500",
        linestyle="--",
        label="Warning threshold"
    )

    plt.axhline(
        digital_twin.critical_threshold * 100,
        color="#DC3545",
        linestyle="--",
        label="Maintenance threshold"
    )

    maintenance_rows = history_df[
        history_df["Event"] == "Maintenance"
    ]

    for _, row in maintenance_rows.iterrows():
        plt.axvline(
            row["Sequence"],
            color="#28A745",
            linestyle=":",
            linewidth=2,
            label=(
                "Maintenance action"
                if _ == maintenance_rows.index[0]
                else None
            )
        )

    plt.title(
        "Closed-Loop Digital Twin: "
        "Failure Risk Before and After Maintenance"
    )
    plt.xlabel("Feedback-loop sequence")
    plt.ylabel("Failure probability (%)")
    plt.ylim(0, 100)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()

    probability_plot_path = (
        RESULTS_FOLDER
        / "feedback_loop_probability.png"
    )

    plt.savefig(
        probability_plot_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # --------------------------------------------------
    # 5. Tool-wear plot
    # --------------------------------------------------

    plt.figure(figsize=(12, 5))

    plt.plot(
        history_df["Sequence"],
        history_df["Tool wear [min]"],
        marker="o",
        linewidth=2.5,
        color="#6F42C1"
    )

    for _, row in maintenance_rows.iterrows():
        plt.axvline(
            row["Sequence"],
            color="#28A745",
            linestyle=":",
            linewidth=2
        )

    plt.title(
        "Tool-Wear Reset Through Maintenance Feedback"
    )
    plt.xlabel("Feedback-loop sequence")
    plt.ylabel("Tool wear [min]")
    plt.grid(alpha=0.25)
    plt.tight_layout()

    wear_plot_path = (
        RESULTS_FOLDER
        / "feedback_loop_tool_wear.png"
    )

    plt.savefig(
        wear_plot_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("\nValidation checks:")

    for check, passed in (
        validation_checks.items()
    ):
        print(
            "PASS" if passed else "FAIL",
            "-",
            check
        )

    print("\nFeedback-loop summary:")
    print(
        "Probability before maintenance:",
        round(
            summary[
                "probability_before_maintenance"
            ],
            4
        )
    )
    print(
        "Probability after maintenance:",
        round(
            summary[
                "probability_after_maintenance"
            ],
            4
        )
    )
    print(
        "Risk reduction:",
        round(
            summary[
                "risk_reduction_percentage_points"
            ],
            2
        ),
        "percentage points"
    )
    print(
        "Final risk level:",
        summary["final_risk_level"]
    )
    print(
        "Overall feedback-loop result:",
        (
            "PASS"
            if feedback_loop_passed
            else "FAIL"
        )
    )

    print("\nHistory saved at:")
    print(history_path)

    print("\nSummary saved at:")
    print(summary_path)

    print("\nPlots saved at:")
    print(probability_plot_path)
    print(wear_plot_path)

    if not feedback_loop_passed:
        raise AssertionError(
            "Closed feedback-loop validation failed."
        )

    print(
        "\nStep 17 complete feedback-loop test passed."
    )
