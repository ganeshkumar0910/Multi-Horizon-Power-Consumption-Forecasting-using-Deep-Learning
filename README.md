# Multi-Horizon Power Consumption Forecasting Using Deep Learning

## Overview

Multi-Horizon Power Consumption Forecasting Using Deep Learning is a production-ready deep learning project designed to forecast electricity consumption in the PJM East Region across multiple forecasting horizons.

The project utilizes a **Multivariate Long Short-Term Memory (LSTM)** architecture with extensive feature engineering to accurately predict future electricity demand for:

- Next Hour
- Next Day
- Next Week
- Next Month

An interactive **Streamlit AI Dashboard** enables users to upload historical sequence data, generate forecasts, visualize predictions, and export professional reports.

---

## Project Features

- **Multivariate LSTM Forecasting**
  - Utilizes 14 engineered features instead of only the target variable, improving forecasting accuracy.

- **Multiple Forecasting Horizons**
  - Dedicated LSTM models for:
    - Hourly Forecasting
    - Daily Forecasting
    - Weekly Forecasting
    - Monthly Forecasting

- **Advanced Feature Engineering**
  - Lag Features
  - Rolling Statistics
  - Calendar Features

- **Interactive AI Dashboard**
  - Built with Streamlit
  - Modern and user-friendly interface

- **Historical Sequence Prediction**
  - Upload historical sequence CSV files for inference.
  - Mimics real-world sequential forecasting workflows.

- **Interactive Visualization**
  - Plotly-based prediction visualization.

- **Professional Report Export**
  - Export prediction reports in:
    - PDF
    - CSV

- **Production-Ready Pipeline**
  - Modular project structure
  - Reusable preprocessing pipeline
  - Saved trained models and scalers

---

# Technology Stack

| Category | Technology |
|------------|------------------------------|
| Programming Language | Python 3 |
| Deep Learning | TensorFlow, Keras (LSTM) |
| Data Processing | Pandas, NumPy, Scikit-learn |
| Dashboard | Streamlit |
| Visualization | Plotly, Matplotlib, Seaborn |
| Report Generation | FPDF2 |

---

# Project Structure

```text
.
├── Dataset/
│   └── PJMW_hourly.csv                 # Original PJM hourly power consumption dataset
│
├── data/
│   ├── cleaned/                        # Cleaned and feature-engineered datasets
│   ├── processed/                      # Generated LSTM sequence arrays
│   ├── daily.csv                       # Daily aggregated dataset
│   ├── hourly.csv                      # Hourly aggregated dataset
│   ├── monthly.csv                     # Monthly aggregated dataset
│   └── weekly.csv                      # Weekly aggregated dataset
│
├── models/
│   ├── scaler/                         # Saved MinMaxScaler objects
│   ├── hourly_model.keras              # Hourly forecasting model
│   ├── daily_model.keras               # Daily forecasting model
│   ├── weekly_model.keras              # Weekly forecasting model
│   └── monthly_model.keras             # Monthly forecasting model
│
├── samples/                            # Sample CSV files for dashboard testing
│
├── src/
│   ├── create_datasets.py              # Creates datasets for all forecasting horizons
│   ├── eda.py                          # Exploratory Data Analysis
│   ├── feature_engineering.py          # Generates multivariate features
│   ├── generate_samples.py             # Creates sample CSV files
│   ├── sequence_generator.py           # Generates LSTM input sequences
│   ├── model_training.py               # Trains LSTM models
│   ├── prediction.py                   # Prediction pipeline
│   └── utils.py                        # Utility functions
│
├── streamlit/
│   └── app.py                          # Streamlit AI Dashboard
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

# Dataset

The project uses the **PJM East Region Hourly Energy Consumption Dataset**.

Place the raw dataset in:

```text
Dataset/
└── PJMW_hourly.csv
```

All processed datasets are automatically generated during preprocessing.

---

# Engineered Features

The LSTM model is trained using the following 14 multivariate features:

| Feature |
|-----------|
| PJMW_MW (Target) |
| Hour |
| Day |
| Month |
| Day_of_Week |
| Is_Weekend |
| Lag_1 |
| Lag_24 |
| Lag_48 |
| Lag_168 |
| Rolling_Mean_24 |
| Rolling_Std_24 |
| Rolling_Min_24 |
| Rolling_Max_24 |

---

# End-to-End Workflow

```text
Raw PJM Dataset
        │
        ▼
Dataset Aggregation
        │
        ▼
Feature Engineering
        │
        ▼
Sequence Generation
        │
        ▼
Multivariate LSTM Training
        │
        ▼
Saved Models & Scalers
        │
        ▼
Prediction Pipeline
        │
        ▼
Interactive Streamlit Dashboard
        │
        ▼
Power Consumption Forecast
```

---

# Installation

## 1. Clone the Repository

```bash
git clone https://github.com/your-username/multi-horizon-power-consumption-forecasting.git

cd multi-horizon-power-consumption-forecasting
```

---

## 2. Create a Virtual Environment (Recommended)

### Windows

```bash
py -3.11 -m venv venv

venv\Scripts\activate
```

### Linux / macOS

```bash
python3.11 -m venv .venv

source .venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Running the Complete Project

## Step 1 — Generate Aggregated Datasets

Reads:

```text
Dataset/PJMW_hourly.csv
```

and creates:

- hourly.csv
- daily.csv
- weekly.csv
- monthly.csv

Run:

```bash
python -m src.create_datasets
```

---

## Step 2 — Perform Feature Engineering

Creates:

- Lag Features
- Rolling Statistics
- Calendar Features

Run:

```bash
python -m src.feature_engineering
```

---

## Step 3 — Generate LSTM Sequences

Creates:

```text
data/processed/
```

containing:

- X_train.npy
- X_test.npy
- y_train.npy
- y_test.npy

for each forecasting horizon.

Run:

```bash
python -m src.sequence_generator
```

---

## Step 4 — Train the Models

Train all four multivariate LSTM models.

```bash
python -m src.model_training
```

Generated models:

```text
models/
├── hourly_model.keras
├── daily_model.keras
├── weekly_model.keras
└── monthly_model.keras
```

Saved scalers:

```text
models/scaler/
```

---

## Step 5 — Generate Sample CSV Files

Generate sample historical sequences for testing.

```bash
python -m src.generate_samples
```

Generated inside:

```text
samples/
```

---

## Step 6 — Launch the AI Dashboard

```bash
streamlit run streamlit/app.py
```

Open your browser:

```
http://localhost:8501
```

---

# Dashboard Usage

## Select Forecast Horizon

Choose one of:

- Hour
- Day
- Week
- Month

---

## Upload Historical Data

Upload a properly formatted historical sequence CSV.

Sample CSV files are available in:

```text
samples/
```

---

## Generate Forecast

The application automatically:

- Loads the trained model
- Loads the corresponding scaler
- Scales the uploaded data
- Creates the LSTM tensor
- Predicts future power consumption
- Displays the forecast

---

## Visualize Results

Interactive Plotly graphs display:

- Historical sequence
- Predicted value

---

## Export Reports

Download prediction reports as:

- PDF
- CSV

---

# Expected Output

The dashboard provides:

- Forecasted Power Consumption
- Interactive Time-Series Visualization
- Historical Sequence Display
- PDF Report
- CSV Report

---

# Future Improvements

- Transformer-based forecasting models
- Hyperparameter optimization
- Weather-aware forecasting
- Real-time prediction API
- Docker support
- Cloud deployment
- Multi-region forecasting

---

# Author

**THUGU GANESH KUMAR REDDY**
