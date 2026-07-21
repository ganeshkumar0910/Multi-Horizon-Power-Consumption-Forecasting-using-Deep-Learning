# Multi-Horizon Power Consumption Forecasting using Deep Learning

This project is a complete, production-grade Deep Learning pipeline designed to forecast electricity consumption for the PJM East Region across multiple time horizons (Next Hour, Next Day, Next Week, Next Month).

It features a **Multivariate Long Short-Term Memory (LSTM)** architecture and a modern **Streamlit AI Dashboard** that supports true historical sequence inference via CSV uploads.

---

##  Project Features

- **Multivariate Forecasting:** The model processes 14 distinct engineered features simultaneously, including lag patterns, rolling statistics, and time components, providing superior accuracy over simple univariate models.
- **Multiple Time Horizons:** Predict power demand for the next Hour, Day, Week, or Month using dedicated models tailored for specific sequence lengths.
- **Production AI Dashboard:** A stunning, recruiter-friendly Streamlit interface mimicking real-world AI applications.
- **Sequence CSV Uploads:** True to how sequential models operate, inference is performed by uploading a sequence of historical data rather than inputting an arbitrary single row.
- **Interactive Visualization:** Leverage Plotly to seamlessly visualize the connection between uploaded historical sequences and the AI's predicted forecast.
- **Exportable Reports:** Download highly professional prediction reports in both PDF (via `fpdf2`) and CSV formats.

---

## Architecture & Tools

| Category | Technology Stack |
| :--- | :--- |
| **Language** | Python 3 |
| **Deep Learning** | TensorFlow / Keras (LSTM) |
| **Data Processing** | Pandas, NumPy, Scikit-learn |
| **Frontend/UI** | Streamlit |
| **Visualization** | Plotly, Matplotlib, Seaborn |
| **Reporting** | FPDF2 |

---

## 📂 Project Structure

```text
├── data/
│   ├── cleaned/             # Preprocessed raw data
│   ├── processed/           # Numpy sequence arrays (X_train, y_train, etc.)
│   └── hourly.csv, daily.csv, etc. # Aggregated datasets
├── models/
│   ├── scaler/              # Saved MinMaxScalers for each horizon
│   └── *_model.keras        # Saved Trained Multivariate LSTM models
├── samples/                 # Automatically generated sample CSVs for testing the UI
├── src/
│   ├── create_datasets.py   # Aggregates raw data into time horizons
│   ├── eda.py               # Exploratory Data Analysis
│   ├── feature_engineering.py # Generates 14 multivariate features (Lags, Rolling stats)
│   ├── generate_samples.py  # Creates the sample CSV files
│   ├── sequence_generator.py # Converts features into 3D LSTM Tensors
│   ├── model_training.py    # Trains the Multivariate LSTM models
│   ├── prediction.py        # Core prediction inference logic
│   └── utils.py             # Reusable pipeline functions (Validation, Scaling)
├── streamlit/
│   └── app.py               # The main Streamlit AI Dashboard
├── requirements.txt         # Project dependencies
├── .gitignore               # Git ignore rules
└── README.md                # This file
```

---

##  The 14 Multivariate Features

To provide the LSTM with maximum context, the following 14 features are engineered and utilized during both training and prediction:

1. `PJMW_MW` (Target)
2. `Hour`
3. `Day`
4. `Month`
5. `Day_of_Week`
6. `Is_Weekend`
7. `Lag_1`
8. `Lag_24`
9. `Lag_48`
10. `Lag_168`
11. `Rolling_Mean_24`
12. `Rolling_Std_24`
13. `Rolling_Min_24`
14. `Rolling_Max_24`

---

## 🚀 How to Run the Project

### 1. Install Dependencies
Make sure you have Python installed, then run:
```bash
pip install -r requirements.txt
```

### 2. Generate Datasets & Sequences
If you are starting from scratch, run the data pipelines:
```bash
python -m src.create_datasets
python -m src.sequence_generator
```

### 3. Train the Models
Train the multivariate LSTM models for all four horizons:
```bash
python -m src.model_training
```

### 4. Launch the AI Dashboard
Start the interactive Streamlit application:
```bash
streamlit run streamlit/app.py
```

---

##  How to use the Prediction Dashboard

1. **Select a Horizon:** Choose between Hour, Day, Week, or Month on the sidebar.
2. **Download a Sample:** In the sidebar, click the download button to grab a perfectly formatted sample CSV sequence for your chosen horizon.
3. **Upload the Data:** Drag and drop the downloaded CSV into the file uploader.
4. **Generate Forecast:** Click the button to watch the AI scale the data, reshape the tensor, predict the future value, and plot the result on an interactive graph.
5. **Download Reports:** Export the results as a PDF or CSV report!

---

##  Developed By

**THUGU GANESH KUMAR REDDY**


