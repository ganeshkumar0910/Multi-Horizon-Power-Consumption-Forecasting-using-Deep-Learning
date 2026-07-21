# =============================================================================
# Project: Multi-Horizon Power Consumption Forecasting using Deep Learning
# File: streamlit/app.py
# Description: Modern, professional AI dashboard for multivariate power forecasting
#              using sequence CSV uploads.
# =============================================================================
import os
import sys
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import plotly.io as pio
import datetime
import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from tensorflow.keras.models import load_model
from fpdf import FPDF
from src.utils import (
    FEATURE_COLUMNS,
    validate_uploaded_csv,
    preprocess_uploaded_data,
    create_lstm_input,
    predict_power,
    inverse_scale_prediction,
    scale_features
)

# 1. Set Page Configuration
st.set_page_config(page_title="AI Power Forecaster", layout="wide")

# 2. Custom CSS
st.markdown("""
<style>
    .badge {
        display: inline-block;
        padding: 0.25em 0.75em;
        font-size: 0.85em;
        font-weight: 600;
        line-height: 1;
        text-align: center;
        white-space: nowrap;
        vertical-align: baseline;
        border-radius: 0.375rem;
        margin-right: 0.5rem;
    }
    .badge-lstm { background-color: #4e73df; color: white; }
    .badge-dl { background-color: #1cc88a; color: white; }
    .badge-power { background-color: #f6c23e; color: black; }
    
    .prediction-card {
        background: linear-gradient(135deg, #1cc88a 0%, #13855c 100%);
        border-radius: 15px;
        padding: 30px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 20px 0;
    }
    .prediction-value {
        font-size: 3.5rem;
        font-weight: bold;
        margin: 10px 0;
    }
    .prediction-label {
        font-size: 1.2rem;
        opacity: 0.9;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .prediction-footer {
        font-size: 0.9rem;
        opacity: 0.8;
        margin-top: 15px;
    }
    
    div[data-testid="stMetric"] {
        background-color: #f8f9fc;
        border-left: 4px solid #4e73df;
        padding: 15px;
        border-radius: 5px;
        box-shadow: 0 0.15rem 1.75rem 0 rgba(58, 59, 69, 0.15);
    }
    
    .footer {
        text-align: center;
        margin-top: 50px;
        padding: 20px;
        color: #858796;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# CACHED FUNCTIONS
# =============================================================================

@st.cache_resource
def load_prediction_model(dataset_type):
    model_path = os.path.join("models", f"{dataset_type}_model.keras")
    if not os.path.exists(model_path):
        return None
    return load_model(model_path)

@st.cache_data
def get_scaler_filename(dataset_type):
    if dataset_type == 'day':
        return "scaler_daily.pkl"
    return f"scaler_{dataset_type}ly.pkl"

@st.cache_data
def load_testing_data(dataset_type):
    data_dir = os.path.join("data", "processed", dataset_type)
    if not os.path.exists(data_dir):
        return None, None
    try:
        X_test = np.load(os.path.join(data_dir, "X_test.npy"))
        y_test = np.load(os.path.join(data_dir, "y_test.npy"))
        return X_test, y_test
    except FileNotFoundError:
        return None, None

def generate_pdf_report(mw_pred, horizon, model_name, seq_len, graph_path):

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    pdf.add_page()

    # -----------------------------
    # Title
    # -----------------------------
    pdf.set_font("Arial", "B", 18)
    pdf.cell(0, 12,
             "Power Consumption Forecast Report",
             ln=True,
             align="C")

    pdf.ln(5)

    pdf.set_font("Arial", size=12)

    pdf.cell(
        0,
        8,
        f"Generated : {datetime.datetime.now().strftime('%d-%m-%Y %H:%M:%S')}",
        ln=True,
    )

    pdf.cell(0, 8, f"Forecast Horizon : {horizon}", ln=True)

    pdf.cell(0, 8, f"Model : {model_name.capitalize()} LSTM", ln=True)

    pdf.cell(0, 8, f"Sequence Length : {seq_len}", ln=True)

    pdf.cell(
        0,
        8,
        f"Predicted Power Consumption : {mw_pred:.2f} MW",
        ln=True,
    )

    pdf.ln(8)

    # -----------------------------
    # Graph
    # -----------------------------
    pdf.set_font("Arial", "B", 14)

    pdf.cell(
        0,
        10,
        "Historical Input and Forecast",
        ln=True,
    )

    if os.path.exists(graph_path):

        pdf.image(
            graph_path,
            x=15,
            w=180
        )

    pdf.ln(105)

    # -----------------------------
    # Summary
    # -----------------------------
    pdf.set_font("Arial", "B", 14)

    pdf.cell(
        0,
        10,
        "Prediction Summary",
        ln=True,
    )

    pdf.set_font("Arial", size=12)

    pdf.multi_cell(
        0,
        8,
        f"""
The uploaded historical sequence was validated successfully.

The trained Multivariate LSTM model analyzed the historical trend
and predicted the next power consumption value.

Predicted Value:

{mw_pred:.2f} MW
"""
    )

    pdf.ln(5)

    pdf.set_font("Arial", "I", 10)

    pdf.cell(
        0,
        10,
        "Generated by Multi-Horizon Power Consumption Forecasting using Deep Learning",
        align="C"
    )

    return bytes(pdf.output())


# =============================================================================
# MAIN STREAMLIT APP
# =============================================================================

def main():
    # --- HEADER ---
    st.title("Multi-Horizon Power Consumption Forecasting")
    st.markdown("""
        <span class="badge badge-lstm">Multivariate LSTM</span>
        <span class="badge badge-dl">Deep Learning</span>
        <span class="badge badge-power">Sequence Inference</span>
    """, unsafe_allow_html=True)
    st.markdown("A professional AI forecasting system. Upload historical sequence data to predict future demand.")
    st.divider()

    # --- SIDEBAR ---
    with st.sidebar:
        st.header("Configuration")
        
        forecast_option = st.selectbox(
            "Forecast Horizon",
            ["Next Hour", "Next Day", "Next Week", "Next Month"]
        )
        
        horizon_map = {
            "Next Hour": "hour",
            "Next Day": "day",
            "Next Week": "week",
            "Next Month": "month"
        }
        dataset_type = horizon_map[forecast_option]
        
        seq_length_map = {
            "hour": 24,
            "day": 30,
            "week": 4,
            "month": 12
        }
        req_seq_len = seq_length_map[dataset_type]
        
        st.divider()
        st.subheader("📊 Model Information")
        st.info(f"**Loaded Model:** {dataset_type.capitalize()} LSTM\n\n**Required Sequence Length:** {req_seq_len} rows\n\n**Input Features:** 14 columns")
        
        st.divider()
        st.subheader("📥 Sample Datasets")
        st.write("Download properly formatted sequences to test the prediction pipeline.")
        
        sample_path = os.path.join("samples", f"sample_{dataset_type}.csv")
        if os.path.exists(sample_path):
            with open(sample_path, "rb") as file:
                st.download_button(
                    label=f"Download {forecast_option} Sample",
                    data=file,
                    file_name=f"sample_{dataset_type}.csv",
                    mime="text/csv"
                )
        else:
            st.warning("Sample dataset not found on disk.")

    # Load Model
    model = load_prediction_model(dataset_type)

    if model is None:
        st.error(f"Required model for '{forecast_option}' is missing. Please train the multivariate model first.")
        return

    # --- TABS ---
    tab1, tab2, tab3, tab4 = st.tabs([
        "Predict Next Timestep", 
        "📈 Model Performance", 
        "📊 Data Visualization", 
        "About Project"
    ])

    # ==========================
    # TAB 1: PREDICTION PIPELINE
    # ==========================
    with tab1:
        st.header("LSTM Sequence Inference")
        st.write("Upload a CSV file containing historical power metrics. The LSTM will analyze the sequence and forecast the next time step.")
        
        uploaded_file = st.file_uploader("Upload Historical Data (CSV)", type="csv")
        
        if uploaded_file is not None:
            # Load Data Preview
            try:
                df = pd.read_csv(uploaded_file)
                st.subheader("Data Preview")
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Dataset Rows", df.shape[0])
                col2.metric("Dataset Columns", df.shape[1])
                col3.metric("Missing Values", int(df.isnull().sum().sum()))
                
                st.dataframe(df.head(), use_container_width=True)
                
                # Validation
                is_valid, val_msg = validate_uploaded_csv(df, req_seq_len)
                
                if not is_valid:
                    st.error(f"Validation Failed: {val_msg}")
                else:
                    st.success("Validation Successful! The sequence format matches the trained LSTM architecture.")
                    
                    if st.button("Generate Forecast", type="primary"):
                        with st.spinner("Processing multivariate sequence..."):
                            try:
                                # Preprocess Data
                                scaler_filename = get_scaler_filename(dataset_type)
                                scaled_seq, scaler = preprocess_uploaded_data(df, scaler_filename)
                                
                                # Create LSTM Tensor
                                lstm_input = create_lstm_input(scaled_seq)
                                
                                # Predict
                                scaled_prediction = predict_power(model, lstm_input)
                                final_mw = inverse_scale_prediction(scaled_prediction, scaler)
                                
                                prediction_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                
                                # Output Card
                                st.markdown(f"""
                                <div class="prediction-card">
                                    <div class="prediction-label">Predicted Power Consumption</div>
                                    <div class="prediction-value">{final_mw:,.2f} MW</div>
                                    <div class="prediction-footer">
                                        <strong>Horizon:</strong> {forecast_option} | <strong>Model:</strong> Multivariate {dataset_type.capitalize()} LSTM<br>
                                        <strong>Sequence Length:</strong> {req_seq_len} | <strong>Features:</strong> 14<br>
                                        <strong>Prediction Time:</strong> {prediction_time}<br><br>
                                        <span style="background: rgba(255,255,255,0.2); padding: 5px 15px; border-radius: 20px;">✓ Inference Successful</span>
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Visualization: Connect Historical to Prediction
                                st.subheader("Forecast Trend Analysis")
                                historical_mw = df['PJMW_MW'].values
                                
                                # Create indices for plotting
                                hist_indices = list(range(1, len(historical_mw) + 1))
                                pred_index = len(historical_mw) + 1
                                
                                fig = go.Figure()
                                # Historical Line
                                fig.add_trace(go.Scatter(x=hist_indices, y=historical_mw, mode='lines+markers', name='Historical Input', line=dict(color='#4e73df', width=2)))
                                # Highlight Last Historical Point
                                fig.add_trace(go.Scatter(x=[hist_indices[-1]], y=[historical_mw[-1]], mode='markers', name='Current State', marker=dict(color='orange', size=10)))
                                # Highlight Predicted Point (connected from last historical)
                                fig.add_trace(go.Scatter(x=[hist_indices[-1], pred_index], y=[historical_mw[-1], final_mw], mode='lines+markers', name='Predicted Forecast', line=dict(color='#1cc88a', width=3, dash='dash'), marker=dict(color='#1cc88a', size=12)))
                                
                                fig.update_layout(title="Sequence Trend -> AI Forecast", xaxis_title="Sequence Step", yaxis_title="Power Consumption (MW)", hovermode="x unified")
                                st.plotly_chart(fig, use_container_width=True)
                                graph_path = "forecast_plot.png"

                                pio.write_image(
                                    fig,
                                    graph_path,
                                    width=900,
                                    height=500
                                )
                                
                                # Download Reports
                                st.subheader("Export Results")
                                
                                # CSV Report
                                report_df = pd.DataFrame({
                                    "Date_Time": [prediction_time],
                                    "Forecast_Horizon": [forecast_option],
                                    "Model": [f"{dataset_type.capitalize()} LSTM"],
                                    "Predicted_MW": [final_mw]
                                })
                                csv_data = report_df.to_csv(index=False).encode('utf-8')
                                
                                # PDF Report
                                pdf_data = generate_pdf_report(
                                    final_mw,
                                    forecast_option,
                                    dataset_type,
                                    req_seq_len,
                                    graph_path
                                )
                                
                                dl_col1, dl_col2, dl_col3 = st.columns(3)
                                with dl_col1:
                                    st.download_button("Download CSV Report", data=csv_data, file_name=f"forecast_report_{dataset_type}.csv", mime="text/csv")
                                with dl_col2:
                                    st.download_button("Download PDF Report", data=pdf_data, file_name=f"forecast_report_{dataset_type}.pdf", mime="application/pdf")
                                
                            except Exception as e:
                                st.error(f"Inference Pipeline Error: {str(e)}")
            except Exception as e:
                st.error(f"Could not parse CSV: {e}")
        else:
            st.info("Awaiting CSV sequence upload...")

    # ==========================
    # TAB 2: MODEL PERFORMANCE
    # ==========================
    with tab2:
        st.header("Evaluation Metrics")
        
        X_test, y_test = load_testing_data(dataset_type)
        if X_test is not None:
            try:
                raw_predictions = model.predict(X_test, verbose=0)
                
                scaler_filename = get_scaler_filename(dataset_type)
                _, scaler = scale_features(np.zeros((1, 14)), scaler_filename, fit=False) 
                
                dummy_y = np.zeros((len(y_test), 14))
                dummy_y[:, 0] = y_test
                actuals = scaler.inverse_transform(dummy_y)[:, 0]
                
                dummy_p = np.zeros((len(raw_predictions), 14))
                dummy_p[:, 0] = raw_predictions.flatten()
                predictions = scaler.inverse_transform(dummy_p)[:, 0]
                
                mae = mean_absolute_error(actuals, predictions)
                mse = mean_squared_error(actuals, predictions)
                rmse = np.sqrt(mse)
                r2 = r2_score(actuals, predictions)
                mape = mean_absolute_percentage_error(actuals, predictions)
                
                col1, col2, col3, col4, col5 = st.columns(5)
                col1.metric("MAE (MW)", f"{mae:,.2f}")
                col2.metric("RMSE (MW)", f"{rmse:,.2f}")
                col3.metric("MSE", f"{mse:,.2f}")
                col4.metric("R² Score", f"{r2:.4f}")
                col5.metric("MAPE", f"{mape * 100:.2f}%")
                
                st.divider()
                st.subheader("Performance Visualization")
                
                plot_df = pd.DataFrame({"Actual": actuals, "Predicted": predictions}).head(200)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['Actual'], mode='lines', name='Actual', line=dict(color='#4e73df', width=2)))
                fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['Predicted'], mode='lines', name='Predicted', line=dict(color='#e74a3b', dash='dash', width=2)))
                fig.update_layout(title=f"Actual vs Predicted - {forecast_option}", xaxis_title="Test Index", yaxis_title="MW", hovermode="x unified")
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.warning(f"Could not calculate metrics: {e}")
        else:
            st.info("Test data not available to display metrics.")

    # ==========================
    # TAB 3: DATA VISUALIZATION
    # ==========================
    with tab3:
        st.header("Interactive Data Exploration")
        if X_test is not None and 'actuals' in locals():
            full_fig = go.Figure()
            full_fig.add_trace(go.Scatter(y=actuals, mode='lines', name='Actual', line=dict(color='royalblue', width=1)))
            full_fig.add_trace(go.Scatter(y=predictions, mode='lines', name='Predicted', line=dict(color='firebrick', width=1, dash='dot')))
            full_fig.update_layout(title="Complete Test Set Forecast Comparison", xaxis_title="Test Index", yaxis_title="MW", hovermode="x unified", dragmode="zoom")
            st.plotly_chart(full_fig, use_container_width=True)
        else:
            st.info("Test data not available.")

    # ==========================
    # TAB 4: ABOUT PROJECT
    # ==========================
    with tab4:
        st.header("Project Architecture & Details")
        
        with st.expander("🎯 Project Objective", expanded=True):
            st.write("This project utilizes Deep Learning to forecast electricity consumption for the PJM East Region. Accurate forecasting is critical for grid stability, preventing blackouts, and optimizing power generation schedules.")
            
        with st.expander("📊 Dataset Description"):
            st.write("**Dataset:** PJME Hourly Energy Consumption Dataset")
            st.write("It contains historical electricity consumption values recorded for the PJM East Region, measured in Megawatts (MW).")
            
        with st.expander("🧠 Multivariate LSTM Architecture"):
            st.write("The core intelligence of the model relies on a Long Short-Term Memory (LSTM) network processing 14 concurrent time-series features.")
            st.write("- **Input Features:** 14 distinct signals including time dependencies, statistical rollings, and lagged values.")
            st.write("- **Input Layer:** Accepts 3D sequence tensors `(samples, sequence_length, 14)`")
            st.write("- **LSTM Layer:** 64 Units for extracting temporal dependencies")
            st.write("- **Dropout Layer:** 20% dropout rate to prevent overfitting")
            st.write("- **Dense Layer:** 32 Units for feature processing")
            st.write("- **Output Layer:** 1 Unit predicting next-step PJMW_MW")
            
        with st.expander("💻 Technologies Used"):
            cols = st.columns(4)
            cols[0].markdown("- Python\n- TensorFlow/Keras")
            cols[1].markdown("- Streamlit\n- Plotly")
            cols[2].markdown("- Scikit-learn\n- Pandas")
            cols[3].markdown("- FPDF2\n- Joblib")

    # --- FOOTER ---
    st.markdown("""
        <div class="footer">
            <hr>
            <strong>Developer:</strong> THUGU GANESH KUMAR REDDY &nbsp; | &nbsp; 
            <strong>Project:</strong> Multivariate Power Forecasting &nbsp; | &nbsp; 
            <strong>Year:</strong> 2026
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
