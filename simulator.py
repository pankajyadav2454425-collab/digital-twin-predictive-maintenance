
from pathlib import Path
from datetime import datetime, timezone
import joblib
import numpy as np
import pandas as pd


class MachineSimulator:

    FEATURE_COLUMNS = [
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

    WEAR_INCREMENT = {
        "L": 2.0,
        "M": 3.0,
        "H": 5.0
    }

    def __init__(
        self,
        data_path,
        model_path,
        initial_product_type="L",
        random_seed=42
    ):

        self.data_path = Path(data_path)
        self.model_path = Path(model_path)

        if not self.data_path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {self.data_path}"
            )

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {self.model_path}"
            )

        if initial_product_type not in {"L", "M", "H"}:
            raise ValueError(
                "Product type must be L, M or H."
            )

        self.rng = np.random.default_rng(random_seed)

        self.data = pd.read_csv(self.data_path)

        self.normal_data = self.data[
            self.data["Machine failure"] == 0
        ].reset_index(drop=True)

        self.failure_data = self.data[
            self.data["Machine failure"] == 1
        ].reset_index(drop=True)

        self.model = joblib.load(self.model_path)

        self.initial_product_type = initial_product_type

        self.history = []
        self.cycle = 0
        self.maintenance_count = 0

        self._initialize_healthy_state()

    def _select_random_row(self, dataframe):

        position = int(
            self.rng.integers(
                low=0,
                high=len(dataframe)
            )
        )

        return dataframe.iloc[position]

    def _initialize_healthy_state(self):

        type_data = self.normal_data[
            self.normal_data["Type"]
            == self.initial_product_type
        ]

        if len(type_data) == 0:
            type_data = self.normal_data

        row = self._select_random_row(type_data)

        self.state = {
            "Type": self.initial_product_type,
            "Air temperature [K]": float(
                row["Air temperature [K]"]
            ),
            "Process temperature [K]": float(
                row["Process temperature [K]"]
            ),
            "Rotational speed [rpm]": int(
                row["Rotational speed [rpm]"]
            ),
            "Torque [Nm]": float(
                row["Torque [Nm]"]
            ),
            "Tool wear [min]": 0.0
        }

    def _select_reference(self, scenario):

        scenario = scenario.lower()

        if scenario == "normal":
            reference_data = self.normal_data
            blend_weight = 0.25

        elif scenario == "mixed":
            reference_data = self.data
            blend_weight = 0.40

        elif scenario == "stress":
            reference_data = self.failure_data
            blend_weight = 0.90

        else:
            raise ValueError(
                "Scenario must be normal, mixed or stress."
            )

        reference = self._select_random_row(
            reference_data
        )

        return reference, blend_weight

    def _calculate_engineered_features(self):

        air_temperature = self.state[
            "Air temperature [K]"
        ]

        process_temperature = self.state[
            "Process temperature [K]"
        ]

        rotational_speed = self.state[
            "Rotational speed [rpm]"
        ]

        torque = self.state["Torque [Nm]"]

        tool_wear = self.state["Tool wear [min]"]

        temperature_difference = (
            process_temperature - air_temperature
        )

        power = (
            torque
            * (
                2
                * np.pi
                * rotational_speed
                / 60
            )
        )

        overstrain_indicator = tool_wear * torque

        return {
            "Temperature difference [K]": float(
                temperature_difference
            ),
            "Power [W]": float(power),
            "Overstrain indicator": float(
                overstrain_indicator
            )
        }

    def _create_model_input(self):

        engineered_features = (
            self._calculate_engineered_features()
        )

        model_input = {
            **self.state,
            **engineered_features
        }

        return pd.DataFrame(
            [model_input],
            columns=self.FEATURE_COLUMNS
        )

    def _predict_current_state(self):

        model_input = self._create_model_input()

        probability = float(
            self.model.predict_proba(
                model_input
            )[0, 1]
        )

        prediction = int(probability >= 0.50)

        return probability, prediction

    def _record_event(
        self,
        scenario,
        event,
        probability,
        prediction
    ):

        engineered_features = (
            self._calculate_engineered_features()
        )

        record = {
            "Timestamp": datetime.now(
                timezone.utc
            ).isoformat(),
            "Cycle": self.cycle,
            "Scenario": scenario,
            "Event": event,
            **self.state,
            **engineered_features,
            "Failure probability": probability,
            "Predicted failure": prediction,
            "Maintenance count": (
                self.maintenance_count
            )
        }

        self.history.append(record)

        return record.copy()

    def step(self, scenario="mixed"):

        reference, blend_weight = (
            self._select_reference(scenario)
        )

        current_weight = 1.0 - blend_weight

        self.state["Type"] = str(
            reference["Type"]
        )

        air_noise = self.rng.normal(
            loc=0.0,
            scale=0.10
        )

        process_noise = self.rng.normal(
            loc=0.0,
            scale=0.10
        )

        speed_noise = self.rng.normal(
            loc=0.0,
            scale=10.0
        )

        torque_noise = self.rng.normal(
            loc=0.0,
            scale=0.50
        )

        self.state["Air temperature [K]"] = float(
            current_weight
            * self.state["Air temperature [K]"]
            + blend_weight
            * float(reference["Air temperature [K]"])
            + air_noise
        )

        self.state["Process temperature [K]"] = float(
            current_weight
            * self.state["Process temperature [K]"]
            + blend_weight
            * float(
                reference[
                    "Process temperature [K]"
                ]
            )
            + process_noise
        )

        updated_speed = (
            current_weight
            * self.state["Rotational speed [rpm]"]
            + blend_weight
            * float(
                reference[
                    "Rotational speed [rpm]"
                ]
            )
            + speed_noise
        )

        self.state["Rotational speed [rpm]"] = int(
            np.clip(
                round(updated_speed),
                1000,
                3000
            )
        )

        updated_torque = (
            current_weight
            * self.state["Torque [Nm]"]
            + blend_weight
            * float(reference["Torque [Nm]"])
            + torque_noise
        )

        self.state["Torque [Nm]"] = float(
            np.clip(
                updated_torque,
                1.0,
                80.0
            )
        )

        wear_increment = self.WEAR_INCREMENT[
            self.state["Type"]
        ]

        self.state["Tool wear [min]"] = float(
            np.clip(
                self.state["Tool wear [min]"]
                + wear_increment,
                0.0,
                250.0
            )
        )

        self.cycle += 1

        probability, prediction = (
            self._predict_current_state()
        )

        return self._record_event(
            scenario=scenario,
            event="Operation",
            probability=probability,
            prediction=prediction
        )

    def perform_maintenance(self):

        current_type = self.state["Type"]

        type_data = self.normal_data[
            self.normal_data["Type"] == current_type
        ]

        if len(type_data) == 0:
            type_data = self.normal_data

        normal_reference = self._select_random_row(
            type_data
        )

        self.state["Air temperature [K]"] = float(
            normal_reference[
                "Air temperature [K]"
            ]
        )

        self.state["Process temperature [K]"] = float(
            normal_reference[
                "Process temperature [K]"
            ]
        )

        self.state["Rotational speed [rpm]"] = int(
            normal_reference[
                "Rotational speed [rpm]"
            ]
        )

        self.state["Torque [Nm]"] = float(
            normal_reference["Torque [Nm]"]
        )

        self.state["Tool wear [min]"] = 0.0

        self.maintenance_count += 1

        probability, prediction = (
            self._predict_current_state()
        )

        return self._record_event(
            scenario="maintenance",
            event="Maintenance",
            probability=probability,
            prediction=prediction
        )

    def reset_simulator(self):

        self.history = []
        self.cycle = 0
        self.maintenance_count = 0

        self._initialize_healthy_state()

        probability, prediction = (
            self._predict_current_state()
        )

        return self._record_event(
            scenario="reset",
            event="Reset",
            probability=probability,
            prediction=prediction
        )

    def get_current_state(self):

        probability, prediction = (
            self._predict_current_state()
        )

        engineered_features = (
            self._calculate_engineered_features()
        )

        return {
            "Cycle": self.cycle,
            **self.state,
            **engineered_features,
            "Failure probability": probability,
            "Predicted failure": prediction,
            "Maintenance count": (
                self.maintenance_count
            )
        }

    def get_history(self):

        return pd.DataFrame(self.history)

    def save_history(self, output_path):

        output_path = Path(output_path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        history_dataframe = self.get_history()

        history_dataframe.to_csv(
            output_path,
            index=False
        )

        return output_path


# --------------------------------------------------
# Test simulator when this file is run directly
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

    history_path = (
        project_root
        / "results"
        / "simulator_test_history.csv"
    )

    simulator = MachineSimulator(
        data_path=data_path,
        model_path=model_path,
        initial_product_type="L",
        random_seed=42
    )

    print("\nMachine simulator started.")
    print("Using calibrated model:", model_path)

    print("\nNormal operating cycles:")

    for _ in range(5):

        state = simulator.step(
            scenario="normal"
        )

        print(
            f"Cycle {state['Cycle']:02d} | "
            f"Wear: {state['Tool wear [min]']:.1f} | "
            f"Speed: "
            f"{state['Rotational speed [rpm]']} | "
            f"Torque: {state['Torque [Nm]']:.2f} | "
            f"Failure probability: "
            f"{state['Failure probability']:.4f}"
        )

    print("\nStress-test cycles:")

    for _ in range(5):

        state = simulator.step(
            scenario="stress"
        )

        print(
            f"Cycle {state['Cycle']:02d} | "
            f"Wear: {state['Tool wear [min]']:.1f} | "
            f"Speed: "
            f"{state['Rotational speed [rpm]']} | "
            f"Torque: {state['Torque [Nm]']:.2f} | "
            f"Failure probability: "
            f"{state['Failure probability']:.4f}"
        )

    before_maintenance = (
        simulator.get_current_state()
    )

    print("\nPerforming maintenance...")

    maintenance_state = (
        simulator.perform_maintenance()
    )

    print(
        "Tool wear before maintenance:",
        round(
            before_maintenance[
                "Tool wear [min]"
            ],
            2
        )
    )

    print(
        "Tool wear after maintenance:",
        round(
            maintenance_state[
                "Tool wear [min]"
            ],
            2
        )
    )

    print(
        "Failure probability after maintenance:",
        round(
            maintenance_state[
                "Failure probability"
            ],
            4
        )
    )

    simulator.save_history(history_path)

    print("\nSimulation history saved at:")
    print(history_path)

    print("\nStep 11 simulator test completed.")
