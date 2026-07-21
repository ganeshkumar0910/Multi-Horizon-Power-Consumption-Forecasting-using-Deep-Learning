# =============================================================================
# Project: Multi-Horizon Power Consumption Forecasting using Deep Learning
# File: src/prediction.py
# Description: Demonstrates how to use the trained multivariate LSTM model 
#              to predict future power consumption using a direct input vector.
# =============================================================================

import os
import numpy as np
from tensorflow.keras.models import load_model
from src.utils import scale_features, predict_power, inverse_scale_prediction

def load_trained_model(dataset_type):
    print(f"Loading Model for {dataset_type} horizon...")
    model_path = os.path.join("models", f"{dataset_type}_model.keras")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    return load_model(model_path)

def make_prediction(dataset_type, user_input_vector):
    """
    Takes a 1D user input vector (14 features), scales it, predicts using 
    the LSTM, and returns the inverse-scaled real Megawatt value.
    """
    print("--- Starting Prediction Pipeline ---")
    
    # 1. Convert to 2D NumPy array for scaling (1 sample, 14 features)
    feature_data = np.array([user_input_vector])
    
    # 2. Scale the features using the saved scaler
    scaler_filename = f"scaler_{dataset_type}ly.pkl"
    if dataset_type == 'day':
        scaler_filename = "scaler_daily.pkl" # Edge case for daily wording
    
    print("Scaling inputs...")
    scaled_data, scaler = scale_features(feature_data, scaler_filename, fit=False)
    
    # 3. Reshape for LSTM: (samples, sequence_length, features) -> (1, 1, 14)
    lstm_input = scaled_data.reshape(1, 1, scaled_data.shape[1])
    
    # 4. Load Model
    model = load_trained_model(dataset_type)
    
    # 5. Predict
    print("Running forecast...")
    scaled_prediction = predict_power(model, lstm_input)
    
    # 6. Inverse Transform
    final_prediction_mw = inverse_scale_prediction(scaled_prediction, scaler)
    
    print(f"\n✅ Forecasted Power Consumption: {final_prediction_mw:,.2f} MW\n")
    return final_prediction_mw

if __name__ == "__main__":
    # Example test run for Hourly Prediction
    # 14 Features: PJMW, Hour, Day, Month, Day_of_Week, Is_Weekend, Lag_1, Lag_24, Lag_48, Lag_168, Roll_Mean, Roll_Std, Roll_Min, Roll_Max
    dummy_input = [5000.0, 12, 15, 6, 2, 0, 4900.0, 4800.0, 4700.0, 4500.0, 4950.0, 100.0, 4000.0, 6000.0]
    
    try:
        make_prediction('hour', dummy_input)
    except Exception as e:
        print(f"Prediction failed: {e}")
