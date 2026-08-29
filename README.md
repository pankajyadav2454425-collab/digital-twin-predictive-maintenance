# Digital Twin Predictive Maintenance System

An explainable, calibrated and cost-aware digital twin for predictive maintenance using the AI4I 2020 dataset.

## Project overview

This system uses machine sensor data to predict calibrated failure probability, classify operational risk and recommend maintenance using expected economic cost.

After maintenance, the digital twin updates the machine state and verifies that the failure risk returns to Healthy.

## Main capabilities

- Calibrated Random Forest failure prediction
- Healthy, Warning and Critical risk classification
- Cost-aware maintenance recommendations
- Machine-state simulation
- Local prediction explanations
- Maintenance-strategy comparison
- Closed-loop maintenance validation
- Interactive Streamlit dashboard

## Model performance

| Metric | Result |
|---|---:|
| Accuracy | 99.30% |
| Failure precision | 96.55% |
| Failure recall | 82.35% |
| Failure F1-score | 88.89% |
| ROC-AUC | 97.60% |
| PR-AUC | 87.77% |

## Economic results

- Failure coverage: 85.29%
- Maintenance precision: 77.33%
- Simulated savings versus corrective maintenance: INR 5.05 million
- Simulated cost reduction: 74.26%

## Closed-loop validation

- Failure probability before maintenance: 98.60%
- Failure probability after maintenance: 0.40%
- Risk changed from Critical to Healthy
- All feedback-loop validation checks passed

## Project structure

```text
digital_twin_predictive_maintenance/
├── app.py
├── simulator.py
├── maintenance_policy.py
├── risk_classifier.py
├── explainability.py
├── compare_strategies.py
├── feedback_loop.py
├── train_model.py
├── calibrate_model.py
├── data/
├── models/
├── results/
├── requirements.txt
└── README.md
```

## Run locally

Install the required libraries:

```bash
pip install -r requirements.txt
```

Start the dashboard:

```bash
streamlit run app.py
```

## Research limitation

The AI4I dataset is a synthetic industrial benchmark. The economic savings reported by this project are simulation results and not production-validated savings.

## Author

**Pankaj Yadav**  
M.Tech, Industrial Engineering & Management  
IIT Kharagpur
