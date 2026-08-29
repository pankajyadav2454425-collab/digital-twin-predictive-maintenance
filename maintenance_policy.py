
from pathlib import Path
import json
import pandas as pd

from simulator import MachineSimulator


class CostAwareMaintenancePolicy:

    def __init__(
        self,
        preventive_maintenance_cost=10000.0,
        failure_cost=100000.0,
        currency="INR"
    ):

        if preventive_maintenance_cost <= 0:
            raise ValueError(
                "Preventive-maintenance cost must be positive."
            )

        if failure_cost <= 0:
            raise ValueError(
                "Failure cost must be positive."
            )

        self.preventive_maintenance_cost = float(
            preventive_maintenance_cost
        )

        self.failure_cost = float(failure_cost)

        self.currency = currency

    @property
    def maintenance_threshold(self):

        return (
            self.preventive_maintenance_cost
            / self.failure_cost
        )

    def evaluate(self, failure_probability):

        failure_probability = float(
            failure_probability
        )

        if not 0.0 <= failure_probability <= 1.0:
            raise ValueError(
                "Failure probability must be "
                "between 0 and 1."
            )

        expected_continue_cost = (
            failure_probability
            * self.failure_cost
        )

        expected_maintenance_cost = (
            self.preventive_maintenance_cost
        )

        perform_maintenance = (
            expected_continue_cost
            >= expected_maintenance_cost
        )

        if perform_maintenance:

            action = "PERFORM MAINTENANCE"

            selected_expected_cost = (
                expected_maintenance_cost
            )

            alternative_expected_cost = (
                expected_continue_cost
            )

            reason = (
                "Expected failure cost is greater than "
                "or equal to preventive-maintenance cost."
            )

        else:

            action = "CONTINUE OPERATION"

            selected_expected_cost = (
                expected_continue_cost
            )

            alternative_expected_cost = (
                expected_maintenance_cost
            )

            reason = (
                "Expected failure cost is lower than "
                "preventive-maintenance cost."
            )

        expected_savings = (
            alternative_expected_cost
            - selected_expected_cost
        )

        return {
            "Failure probability": (
                failure_probability
            ),
            "Maintenance threshold": (
                self.maintenance_threshold
            ),
            "Preventive maintenance cost": (
                expected_maintenance_cost
            ),
            "Failure cost": self.failure_cost,
            "Expected continue cost": (
                expected_continue_cost
            ),
            "Expected maintenance cost": (
                expected_maintenance_cost
            ),
            "Recommended action": action,
            "Perform maintenance": (
                perform_maintenance
            ),
            "Selected expected cost": (
                selected_expected_cost
            ),
            "Alternative expected cost": (
                alternative_expected_cost
            ),
            "Expected savings": (
                expected_savings
            ),
            "Decision reason": reason,
            "Currency": self.currency
        }

    def evaluate_machine_state(self, machine_state):

        if "Failure probability" not in machine_state:
            raise KeyError(
                "Machine state does not contain "
                "'Failure probability'."
            )

        decision = self.evaluate(
            machine_state["Failure probability"]
        )

        return {
            "Cycle": machine_state.get("Cycle"),
            "Scenario": machine_state.get(
                "Scenario"
            ),
            "Event": machine_state.get("Event"),
            "Tool wear [min]": machine_state.get(
                "Tool wear [min]"
            ),
            "Rotational speed [rpm]": (
                machine_state.get(
                    "Rotational speed [rpm]"
                )
            ),
            "Torque [Nm]": machine_state.get(
                "Torque [Nm]"
            ),
            **decision
        }

    def update_costs(
        self,
        preventive_maintenance_cost=None,
        failure_cost=None
    ):

        if preventive_maintenance_cost is not None:

            if preventive_maintenance_cost <= 0:
                raise ValueError(
                    "Preventive-maintenance cost "
                    "must be positive."
                )

            self.preventive_maintenance_cost = float(
                preventive_maintenance_cost
            )

        if failure_cost is not None:

            if failure_cost <= 0:
                raise ValueError(
                    "Failure cost must be positive."
                )

            self.failure_cost = float(failure_cost)

        return {
            "Preventive maintenance cost": (
                self.preventive_maintenance_cost
            ),
            "Failure cost": self.failure_cost,
            "Maintenance threshold": (
                self.maintenance_threshold
            )
        }

    def get_configuration(self):

        return {
            "preventive_maintenance_cost": (
                self.preventive_maintenance_cost
            ),
            "failure_cost": self.failure_cost,
            "maintenance_threshold": (
                self.maintenance_threshold
            ),
            "currency": self.currency
        }


# --------------------------------------------------
# Test policy when file is run directly
# --------------------------------------------------

if __name__ == "__main__":

    project_root = Path(__file__).resolve().parent

    data_path = (
        project_root
        / "data"
        / "ai4i2020.csv"
    )

    model_path = (
        project_root
        / "models"
        / "failure_model_calibrated.pkl"
    )

    results_path = (
        project_root
        / "results"
        / "maintenance_policy_test.csv"
    )

    configuration_path = (
        project_root
        / "results"
        / "maintenance_policy_config.json"
    )

    policy = CostAwareMaintenancePolicy(
        preventive_maintenance_cost=10000,
        failure_cost=100000,
        currency="INR"
    )

    print("\nCost-aware maintenance policy")

    print(
        "Preventive-maintenance cost:",
        policy.preventive_maintenance_cost
    )

    print(
        "Failure cost:",
        policy.failure_cost
    )

    print(
        "Calculated maintenance threshold:",
        round(
            policy.maintenance_threshold,
            4
        )
    )

    print("\nBasic probability tests:")

    test_probabilities = [
        0.01,
        0.05,
        0.10,
        0.25,
        0.95
    ]

    for probability in test_probabilities:

        decision = policy.evaluate(
            probability
        )

        print(
            f"Probability: {probability:.2%} | "
            f"Expected failure cost: "
            f"{decision['Expected continue cost']:.2f} | "
            f"Action: "
            f"{decision['Recommended action']}"
        )

    print("\nTesting policy with machine simulator:")

    simulator = MachineSimulator(
        data_path=data_path,
        model_path=model_path,
        initial_product_type="L",
        random_seed=42
    )

    policy_results = []

    scenarios = (
        ["normal"] * 5
        + ["stress"] * 5
    )

    for scenario in scenarios:

        machine_state = simulator.step(
            scenario=scenario
        )

        decision = policy.evaluate_machine_state(
            machine_state
        )

        policy_results.append(decision)

        print(
            f"Cycle {decision['Cycle']:02d} | "
            f"Scenario: {scenario:6s} | "
            f"Failure probability: "
            f"{decision['Failure probability']:.4f} | "
            f"Threshold: "
            f"{decision['Maintenance threshold']:.4f} | "
            f"Action: "
            f"{decision['Recommended action']}"
        )

    results_dataframe = pd.DataFrame(
        policy_results
    )

    results_dataframe.to_csv(
        results_path,
        index=False
    )

    with open(
        configuration_path,
        "w"
    ) as file:

        json.dump(
            policy.get_configuration(),
            file,
            indent=4
        )

    print("\nPolicy-test results saved at:")
    print(results_path)

    print("\nPolicy configuration saved at:")
    print(configuration_path)

    print("\nStep 12 policy test completed.")
