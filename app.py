
from pathlib import Path
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from maintenance_policy import CostAwareMaintenancePolicy
from risk_classifier import MachineRiskClassifier
from explainability import LocalPredictionExplainer


# --------------------------------------------------
# 1. Configuration and paths
# --------------------------------------------------

st.set_page_config(
    page_title="Digital Twin Predictive Maintenance",
    page_icon="⚙️",
    layout="wide"
)

PROJECT_ROOT = Path(__file__).resolve().parent

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "failure_model_calibrated.pkl"
)

RESULTS_FOLDER = PROJECT_ROOT / "results"

DASHBOARD_HISTORY_PATH = (
    RESULTS_FOLDER / "dashboard_history.csv"
)

RESULTS_FOLDER.mkdir(exist_ok=True)


# --------------------------------------------------
# 2. Session-state initialization
# --------------------------------------------------

session_defaults = {
    "dashboard_history": [],
    "cycle_number": 0,
    "maintenance_count": 0,
    "maintenance_event_pending": False,
    "machine_type_widget": "M",
    "air_temperature_widget": 298.5,
    "process_temperature_widget": 309.1,
    "rotational_speed_widget": 1705,
    "torque_widget": 33.66,
    "tool_wear_widget": 8.0
}

for key, value in session_defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# --------------------------------------------------
# 3. Preset and maintenance callbacks
# --------------------------------------------------

def load_healthy_preset():

    st.session_state["machine_type_widget"] = "M"
    st.session_state["air_temperature_widget"] = 298.5
    st.session_state["process_temperature_widget"] = 309.1
    st.session_state["rotational_speed_widget"] = 1705
    st.session_state["torque_widget"] = 33.66
    st.session_state["tool_wear_widget"] = 8.0


def load_stress_preset():

    st.session_state["machine_type_widget"] = "L"
    st.session_state["air_temperature_widget"] = 299.45
    st.session_state["process_temperature_widget"] = 309.10
    st.session_state["rotational_speed_widget"] = 1259
    st.session_state["torque_widget"] = 68.46
    st.session_state["tool_wear_widget"] = 27.0


def perform_maintenance():

    st.session_state["tool_wear_widget"] = 0.0
    st.session_state["maintenance_count"] += 1
    st.session_state["maintenance_event_pending"] = True


# --------------------------------------------------
# 4. Load calibrated model
# --------------------------------------------------

@st.cache_resource
def load_failure_model(model_path):

    return joblib.load(model_path)


if not MODEL_PATH.exists():
    st.error(
        "Calibrated model not found. "
        "Run calibrate_model.py first."
    )
    st.stop()

model = load_failure_model(str(MODEL_PATH))


@st.cache_resource
def load_local_explainer(
    model_path,
    data_path
):

    return LocalPredictionExplainer(
        model_path=model_path,
        data_path=data_path
    )


local_explainer = load_local_explainer(
    str(MODEL_PATH),
    str(PROJECT_ROOT / "data" / "ai4i2020.csv")
)


# --------------------------------------------------
# 5. Sidebar controls
# --------------------------------------------------

st.sidebar.title("⚙️ Digital Twin Controls")

st.sidebar.subheader("Machine-state presets")

preset_col1, preset_col2 = st.sidebar.columns(2)

preset_col1.button(
    "Healthy preset",
    on_click=load_healthy_preset,
    use_container_width=True
)

preset_col2.button(
    "Stress preset",
    on_click=load_stress_preset,
    use_container_width=True
)

st.sidebar.divider()
st.sidebar.subheader("Machine sensor inputs")

machine_type = st.sidebar.selectbox(
    "Product type",
    options=["L", "M", "H"],
    key="machine_type_widget"
)

air_temperature = st.sidebar.slider(
    "Air temperature [K]",
    min_value=295.0,
    max_value=305.0,
    step=0.1,
    key="air_temperature_widget"
)

process_temperature = st.sidebar.slider(
    "Process temperature [K]",
    min_value=305.0,
    max_value=315.0,
    step=0.1,
    key="process_temperature_widget"
)

rotational_speed = st.sidebar.slider(
    "Rotational speed [rpm]",
    min_value=1100,
    max_value=3000,
    step=1,
    key="rotational_speed_widget"
)

torque = st.sidebar.slider(
    "Torque [Nm]",
    min_value=3.0,
    max_value=80.0,
    step=0.1,
    key="torque_widget"
)

tool_wear = st.sidebar.slider(
    "Tool wear [min]",
    min_value=0.0,
    max_value=250.0,
    step=1.0,
    key="tool_wear_widget"
)

st.sidebar.divider()
st.sidebar.subheader("Maintenance economics")

maintenance_cost = st.sidebar.number_input(
    "Preventive-maintenance cost [₹]",
    min_value=1000.0,
    value=10000.0,
    step=1000.0
)

failure_cost = st.sidebar.number_input(
    "Machine-failure cost [₹]",
    min_value=1000.0,
    value=100000.0,
    step=5000.0
)

if maintenance_cost >= failure_cost:
    st.sidebar.error(
        "Failure cost must be greater than "
        "preventive-maintenance cost."
    )
    st.stop()


# --------------------------------------------------
# 6. Feature engineering
# --------------------------------------------------

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

machine_features = pd.DataFrame(
    {
        "Type": [machine_type],
        "Air temperature [K]": [air_temperature],
        "Process temperature [K]": [
            process_temperature
        ],
        "Rotational speed [rpm]": [
            rotational_speed
        ],
        "Torque [Nm]": [torque],
        "Tool wear [min]": [tool_wear],
        "Temperature difference [K]": [
            temperature_difference
        ],
        "Power [W]": [power],
        "Overstrain indicator": [
            overstrain_indicator
        ]
    }
)


# --------------------------------------------------
# 7. Digital-twin predictions and decisions
# --------------------------------------------------

failure_probability = float(
    model.predict_proba(machine_features)[0, 1]
)

predicted_failure = int(
    model.predict(machine_features)[0]
)

maintenance_policy = CostAwareMaintenancePolicy(
    preventive_maintenance_cost=maintenance_cost,
    failure_cost=failure_cost,
    currency="INR"
)

policy_result = maintenance_policy.evaluate(
    failure_probability
)

critical_threshold = round(
    maintenance_policy.maintenance_threshold,
    10
)

warning_threshold = round(
    critical_threshold * 0.20,
    10
)

risk_classifier = MachineRiskClassifier(
    warning_threshold=warning_threshold,
    critical_threshold=critical_threshold
)

risk_result = risk_classifier.classify(
    failure_probability
)


# --------------------------------------------------
# 8. Dashboard heading and metrics
# --------------------------------------------------

st.title("⚙️ Digital Twin Predictive Maintenance System")

st.caption(
    "AI4I machine-failure prediction using a calibrated "
    "Random Forest, risk classification and cost-aware "
    "maintenance decisions."
)

metric_col1, metric_col2, metric_col3, metric_col4 = (
    st.columns(4)
)

metric_col1.metric(
    "Failure probability",
    f"{failure_probability:.2%}"
)

metric_col2.metric(
    "Machine risk",
    risk_result["Risk level"]
)

metric_col3.metric(
    "Maintenance threshold",
    f"{critical_threshold:.2%}"
)

metric_col4.metric(
    "Maintenance count",
    st.session_state["maintenance_count"]
)

risk_color = risk_result["Risk color"]

st.markdown(
    f"""
    <div style="
        background-color:{risk_color}20;
        border-left:8px solid {risk_color};
        padding:16px;
        border-radius:8px;
        margin-top:10px;
        margin-bottom:20px;
    ">
        <h3 style="color:{risk_color}; margin:0;">
            {risk_result["Risk level"]} machine condition
        </h3>
        <p style="margin-top:8px; margin-bottom:4px;">
            <b>Risk response:</b>
            {risk_result["Risk recommendation"]}
        </p>
        <p style="margin:0;">
            {risk_result["Risk description"]}
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


# --------------------------------------------------
# 9. Probability gauge and maintenance decision
# --------------------------------------------------

gauge_column, decision_column = st.columns([1.2, 1])

gauge_figure = go.Figure(
    go.Indicator(
        mode="gauge+number",
        value=failure_probability * 100,
        number={
            "suffix": "%",
            "valueformat": ".2f"
        },
        title={
            "text": "Predicted failure risk"
        },
        gauge={
            "axis": {
                "range": [0, 100]
            },
            "bar": {
                "color": risk_color
            },
            "steps": [
                {
                    "range": [
                        0,
                        warning_threshold * 100
                    ],
                    "color": "#D8F3DC"
                },
                {
                    "range": [
                        warning_threshold * 100,
                        critical_threshold * 100
                    ],
                    "color": "#FFF3CD"
                },
                {
                    "range": [
                        critical_threshold * 100,
                        100
                    ],
                    "color": "#F8D7DA"
                }
            ],
            "threshold": {
                "line": {
                    "color": "#DC3545",
                    "width": 4
                },
                "value": critical_threshold * 100
            }
        }
    )
)

gauge_figure.update_layout(
    height=390,
    margin=dict(
        l=30,
        r=30,
        t=70,
        b=20
    )
)

gauge_column.plotly_chart(
    gauge_figure,
    use_container_width=True
)

with decision_column:

    st.subheader("Maintenance decision")

    st.metric(
        "Recommended action",
        policy_result["Recommended action"]
    )

    st.metric(
        "Expected cost of continuing",
        f"₹{policy_result['Expected continue cost']:,.0f}"
    )

    st.metric(
        "Preventive-maintenance cost",
        f"₹{maintenance_cost:,.0f}"
    )

    st.metric(
        "Expected savings from selected action",
        f"₹{policy_result['Expected savings']:,.0f}"
    )

    st.info(
        policy_result["Decision reason"]
    )


# --------------------------------------------------
# 10. Current machine-state details
# --------------------------------------------------

with st.expander(
    "View current digital-twin state",
    expanded=False
):

    state_col1, state_col2, state_col3 = st.columns(3)

    state_col1.metric(
        "Temperature difference",
        f"{temperature_difference:.2f} K"
    )

    state_col1.metric(
        "Rotational speed",
        f"{rotational_speed:,} rpm"
    )

    state_col2.metric(
        "Torque",
        f"{torque:.2f} Nm"
    )

    state_col2.metric(
        "Calculated power",
        f"{power:,.2f} W"
    )

    state_col3.metric(
        "Tool wear",
        f"{tool_wear:.1f} min"
    )

    state_col3.metric(
        "Overstrain indicator",
        f"{overstrain_indicator:,.2f}"
    )

    st.write(
        "Model prediction:",
        "Failure" if predicted_failure == 1 else "Normal"
    )


# --------------------------------------------------
# Local individual-prediction explanation
# --------------------------------------------------

st.divider()
st.subheader("🔍 Why did the model make this prediction?")

current_raw_state = {
    "Type": machine_type,
    "Air temperature [K]": air_temperature,
    "Process temperature [K]": process_temperature,
    "Rotational speed [rpm]": rotational_speed,
    "Torque [Nm]": torque,
    "Tool wear [min]": tool_wear
}

local_explanation = local_explainer.explain(
    current_raw_state
)

explanation_plot_data = (
    local_explanation.sort_values(
        "Contribution percentage points"
    )
)

explanation_colors = [
    "#DC3545"
    if contribution > 0
    else "#28A745"
    if contribution < 0
    else "#6C757D"
    for contribution in explanation_plot_data[
        "Contribution percentage points"
    ]
]

explanation_figure = go.Figure(
    go.Bar(
        x=explanation_plot_data[
            "Contribution percentage points"
        ],
        y=explanation_plot_data["Feature"],
        orientation="h",
        marker_color=explanation_colors,
        customdata=explanation_plot_data[
            [
                "Current value",
                "Reference value",
                "Direction"
            ]
        ],
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Contribution: %{x:.3f} percentage points<br>"
            "Current: %{customdata[0]}<br>"
            "Reference: %{customdata[1]}<br>"
            "%{customdata[2]}"
            "<extra></extra>"
        )
    )
)

explanation_figure.add_vline(
    x=0,
    line_color="black",
    line_width=1
)

explanation_figure.update_layout(
    title="Local feature effects on failure risk",
    xaxis_title=(
        "Change in failure probability "
        "(percentage points)"
    ),
    yaxis_title="Feature",
    height=430,
    margin=dict(
        l=30,
        r=30,
        t=60,
        b=30
    )
)

st.plotly_chart(
    explanation_figure,
    use_container_width=True
)

positive_drivers = local_explanation[
    local_explanation[
        "Contribution percentage points"
    ] > 0
]

if not positive_drivers.empty:

    top_driver = positive_drivers.iloc[0]

    st.warning(
        f"Top local risk driver: "
        f"{top_driver['Feature']} "
        f"({top_driver['Contribution percentage points']:.3f} "
        f"percentage points)."
    )

else:
    st.success(
        "No evaluated feature is currently increasing "
        "risk relative to its reference value."
    )

with st.expander(
    "View detailed feature explanation"
):

    st.dataframe(
        local_explanation[
            [
                "Feature",
                "Current value",
                "Reference value",
                "Contribution percentage points",
                "Direction"
            ]
        ],
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "This is a local counterfactual sensitivity "
        "explanation. It compares the current machine "
        "state with typical AI4I reference values. "
        "Contributions are directional and are not "
        "expected to add exactly to the prediction."
    )


# --------------------------------------------------
# 11. Record and maintenance buttons
# --------------------------------------------------

button_col1, button_col2, button_col3 = st.columns(3)

record_clicked = button_col1.button(
    "Record current cycle",
    type="primary",
    use_container_width=True
)

button_col2.button(
    "Perform maintenance",
    on_click=perform_maintenance,
    use_container_width=True
)

clear_clicked = button_col3.button(
    "Clear dashboard history",
    use_container_width=True
)

if clear_clicked:
    st.session_state["dashboard_history"] = []
    st.session_state["cycle_number"] = 0

event_type = None

if record_clicked:
    event_type = "Operation"

if st.session_state.pop(
    "maintenance_event_pending",
    False
):
    event_type = "Maintenance"
    st.success(
        "Maintenance performed: tool wear reset to zero."
    )

if event_type is not None:

    st.session_state["cycle_number"] += 1

    history_row = {
        "Timestamp": datetime.now(
            timezone.utc
        ).isoformat(),
        "Cycle": st.session_state["cycle_number"],
        "Event": event_type,
        "Type": machine_type,
        "Air temperature [K]": air_temperature,
        "Process temperature [K]": process_temperature,
        "Rotational speed [rpm]": rotational_speed,
        "Torque [Nm]": torque,
        "Tool wear [min]": tool_wear,
        "Temperature difference [K]": (
            temperature_difference
        ),
        "Power [W]": power,
        "Overstrain indicator": overstrain_indicator,
        "Failure probability": failure_probability,
        "Predicted failure": predicted_failure,
        "Risk level": risk_result["Risk level"],
        "Recommended action": policy_result[
            "Recommended action"
        ],
        "Expected continue cost": policy_result[
            "Expected continue cost"
        ],
        "Maintenance threshold": critical_threshold,
        "Maintenance count": st.session_state[
            "maintenance_count"
        ]
    }

    st.session_state["dashboard_history"].append(
        history_row
    )


# --------------------------------------------------
# 12. Dashboard history
# --------------------------------------------------

st.divider()
st.subheader("Digital-twin history")

if st.session_state["dashboard_history"]:

    history_df = pd.DataFrame(
        st.session_state["dashboard_history"]
    )

    history_df.to_csv(
        DASHBOARD_HISTORY_PATH,
        index=False
    )

    probability_figure = go.Figure()

    probability_figure.add_trace(
        go.Scatter(
            x=history_df["Cycle"],
            y=history_df["Failure probability"],
            mode="lines",
            name="Failure probability",
            line={
                "color": "#4682B4",
                "width": 3
            }
        )
    )

    risk_colors = {
        "Healthy": "#28A745",
        "Warning": "#FFA500",
        "Critical": "#DC3545"
    }

    for risk_level, color in risk_colors.items():

        risk_rows = history_df[
            history_df["Risk level"] == risk_level
        ]

        if not risk_rows.empty:
            probability_figure.add_trace(
                go.Scatter(
                    x=risk_rows["Cycle"],
                    y=risk_rows[
                        "Failure probability"
                    ],
                    mode="markers",
                    name=risk_level,
                    marker={
                        "color": color,
                        "size": 12,
                        "line": {
                            "color": "black",
                            "width": 1
                        }
                    }
                )
            )

    probability_figure.add_hline(
        y=warning_threshold,
        line_dash="dash",
        line_color="#FFA500",
        annotation_text="Warning threshold"
    )

    probability_figure.add_hline(
        y=critical_threshold,
        line_dash="dash",
        line_color="#DC3545",
        annotation_text="Maintenance threshold"
    )

    probability_figure.update_layout(
        title="Failure probability across cycles",
        xaxis_title="Cycle",
        yaxis_title="Failure probability",
        height=430
    )

    probability_figure.update_yaxes(
        tickformat=".1%",
        range=[0, 1]
    )

    st.plotly_chart(
        probability_figure,
        use_container_width=True
    )

    wear_figure = go.Figure()

    wear_figure.add_trace(
        go.Scatter(
            x=history_df["Cycle"],
            y=history_df["Tool wear [min]"],
            mode="lines+markers",
            name="Tool wear",
            line={
                "color": "#6F42C1",
                "width": 3
            }
        )
    )

    wear_figure.update_layout(
        title="Tool-wear history",
        xaxis_title="Cycle",
        yaxis_title="Tool wear [min]",
        height=350
    )

    st.plotly_chart(
        wear_figure,
        use_container_width=True
    )

    st.dataframe(
        history_df,
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        label="Download dashboard history",
        data=history_df.to_csv(
            index=False
        ).encode("utf-8"),
        file_name="digital_twin_dashboard_history.csv",
        mime="text/csv"
    )

else:
    st.info(
        "Use 'Record current cycle' to begin "
        "the digital-twin history."
    )


# --------------------------------------------------
# Maintenance-strategy comparison
# --------------------------------------------------

strategy_comparison_path = (
    RESULTS_FOLDER
    / "maintenance_strategy_comparison.csv"
)

if strategy_comparison_path.exists():

    st.divider()
    st.subheader("📊 Maintenance strategy comparison")

    strategy_comparison = pd.read_csv(
        strategy_comparison_path
    )

    best_strategy_row = strategy_comparison.loc[
        strategy_comparison[
            "Total cost"
        ].idxmin()
    ]

    strategy_col1, strategy_col2 = st.columns(2)

    strategy_col1.metric(
        "Lowest-cost strategy",
        best_strategy_row["Strategy"]
    )

    strategy_col2.metric(
        "Savings vs corrective",
        f"₹{best_strategy_row['Savings vs corrective']:,.0f}"
    )

    strategy_cost_figure = go.Figure(
        go.Bar(
            x=strategy_comparison["Strategy"],
            y=strategy_comparison["Total cost"],
            marker_color=[
                "#6C757D",
                "#FFA500",
                "#28A745"
            ],
            text=[
                f"₹{value:,.0f}"
                for value in strategy_comparison[
                    "Total cost"
                ]
            ],
            textposition="auto"
        )
    )

    strategy_cost_figure.update_layout(
        title="Total economic cost by strategy",
        xaxis_title="Strategy",
        yaxis_title="Total cost [₹]",
        height=430,
        margin=dict(
            l=30,
            r=30,
            t=60,
            b=30
        )
    )

    st.plotly_chart(
        strategy_cost_figure,
        use_container_width=True
    )

    strategy_display = strategy_comparison[
        [
            "Strategy",
            "Planned maintenance",
            "Prevented failures",
            "Unplanned failures",
            "Failure coverage",
            "Total cost",
            "Savings vs corrective"
        ]
    ].copy()

    strategy_display[
        "Failure coverage"
    ] = (
        strategy_display["Failure coverage"]
        * 100
    ).round(2)

    st.dataframe(
        strategy_display,
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "This comparison is an economic simulation on "
        "the held-out AI4I observations. A planned "
        "maintenance action on a failure-labelled "
        "observation is assumed to prevent that failure."
    )


# --------------------------------------------------
# 13. Footer
# --------------------------------------------------

st.divider()

st.caption(
    "Digital Twin Predictive Maintenance Project | "
    "AI4I 2020 Dataset | Calibrated Random Forest"
)


# --------------------------------------------------
# Developer information
# --------------------------------------------------

st.markdown("---")

st.markdown(
    """
    <div style="
        text-align: center;
        color: #6b7280;
        padding: 12px 0 20px 0;
        font-size: 15px;
    ">
        <strong>Developed by Pankaj Yadav</strong><br>
        M.Tech, Industrial Engineering & Management,
        IIT Kharagpur
    </div>
    """,
    unsafe_allow_html=True,
)
