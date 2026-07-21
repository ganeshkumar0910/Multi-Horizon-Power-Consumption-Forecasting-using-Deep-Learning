import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib

FEATURE_COLUMNS = [
    "PJMW_MW",
    "Hour",
    "Day",
    "Month",
    "Day_of_Week",
    "Is_Weekend",
    "Lag_1",
    "Lag_24",
    "Lag_48",
    "Lag_168",
    "Rolling_Mean_24",
    "Rolling_Std_24",
    "Rolling_Min_24",
    "Rolling_Max_24"
]

def prepare_features(dataset):
    """
    Extracts only the required feature columns for multivariate forecasting.
    Expects a pandas DataFrame. Returns a NumPy array.
    """
    missing_cols = [col for col in FEATURE_COLUMNS if col not in dataset.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in dataset: {missing_cols}")
        
    return dataset[FEATURE_COLUMNS].values


def scale_features(feature_data, scaler_filename, fit=True):
    """
    Scales the multivariate feature data using MinMaxScaler.
    If fit=True, fits a new scaler and saves it.
    If fit=False, loads the existing scaler and transforms.
    """
    scaler_dir = os.path.join("models", "scaler")
    scaler_path = os.path.join(scaler_dir, scaler_filename)
    
    if fit:
        if not os.path.exists(scaler_dir):
            os.makedirs(scaler_dir)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(feature_data)
        joblib.dump(scaler, scaler_path)
    else:
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler not found at {scaler_path}")
        scaler = joblib.load(scaler_path)
        scaled_data = scaler.transform(feature_data)
        
    return scaled_data, scaler


def create_sequences(scaled_data, sequence_length):
    """
    Generates multivariate sequences for LSTM training.
    """
    X = []
    y = []
    
    for i in range(len(scaled_data) - sequence_length):
        sequence_input = scaled_data[i : i + sequence_length]
        sequence_target = scaled_data[i + sequence_length, 0]
        X.append(sequence_input)
        y.append(sequence_target)
        
    return np.array(X), np.array(y)


def predict_power(model, X_input):
    """
    Performs a prediction using the trained LSTM model.
    """
    return model.predict(X_input, verbose=0)


def inverse_scale_prediction(scaled_prediction, scaler):
    """
    Creates a dummy array of 14 features to inverse transform the single predicted PJMW_MW value.
    """
    dummy_row = np.zeros(len(FEATURE_COLUMNS))
    
    if isinstance(scaled_prediction, (list, np.ndarray)):
        val = scaled_prediction[0]
        if isinstance(val, (list, np.ndarray)):
            val = val[0]
    else:
        val = scaled_prediction
        
    dummy_row[0] = val
    inverse_transformed = scaler.inverse_transform([dummy_row])
    return inverse_transformed[0][0]


# =============================================================================
# NEW UI WORKFLOW FUNCTIONS
# =============================================================================

def validate_uploaded_csv(df, expected_sequence_length):
    """
    Validates an uploaded CSV for the prediction pipeline.
    Ensures format, column order, missing values, and row count are correct.
    """
    # 1. Check required columns exist
    missing_cols = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing_cols:
        return False, f"Missing required columns: {', '.join(missing_cols)}"
    
    # 2. Check missing values
    if df[FEATURE_COLUMNS].isnull().values.any():
        return False, "The uploaded file contains missing values (NaN). Please fill them."
        
    # 3. Check correct number of rows
    if len(df) != expected_sequence_length:
        return False, f"Expected exactly {expected_sequence_length} rows for this horizon. Found {len(df)} rows."
        
    return True, "Validation Successful"


def preprocess_uploaded_data(df, scaler_filename):
    """
    Extracts the feature columns in exact order, applies the saved scaler,
    and returns the 2D scaled data and the scaler object.
    """
    # Force exact column order
    ordered_data = df[FEATURE_COLUMNS].values
    
    # Load scaler and transform (fit=False)
    scaled_data, scaler = scale_features(ordered_data, scaler_filename, fit=False)
    
    return scaled_data, scaler


def create_lstm_input(scaled_data):
    """
    Reshapes a 2D array (sequence_length, features) into the 3D tensor
    expected by the LSTM: (1, sequence_length, features).
    """
    seq_len = scaled_data.shape[0]
    num_features = scaled_data.shape[1]
    
    lstm_input = scaled_data.reshape(1, seq_len, num_features)
    return lstm_input
