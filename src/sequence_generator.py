# =============================================================================
# Project: Multi-Horizon Power Consumption Forecasting using Deep Learning
# File: src/sequence_generator.py
# Description: This script converts our multivariate time-series datasets into 
#              supervised learning sequences (X and y) specifically designed 
#              for training an LSTM neural network.
# =============================================================================

import os
import numpy as np
import pandas as pd
from src.utils import prepare_features, scale_features, create_sequences

def load_dataset(file_path):
    """
    Load a specific dataset (hourly, daily, monthly, or weekly).
    """
    print(f"Loading Dataset: {file_path}")
    if not os.path.exists(file_path):
        print("Dataset not found. Please generate datasets first.")
        return None
        
    dataset = pd.read_csv(file_path, index_col='Datetime', parse_dates=True)
    return dataset


def split_data(X, y):
    """
    Split the sequences into Training (80%) and Testing (20%) sets.
    """
    print("Splitting Dataset...")
    split_index = int(len(X) * 0.8)
    
    X_train = X[:split_index]
    y_train = y[:split_index]
    X_test = X[split_index:]
    y_test = y[split_index:]
    
    print(f"Training Shape: {X_train.shape}")
    print(f"Testing Shape: {X_test.shape}")
    print()
    return X_train, X_test, y_train, y_test


def save_sequences(X_train, X_test, y_train, y_test, folder_name):
    """
    Save the generated sequences to the hard drive as fast-loading NumPy files (.npy).
    """
    print("Saving Files...")
    base_folder = os.path.join("data", "processed", folder_name)
    if not os.path.exists(base_folder):
        os.makedirs(base_folder)
        
    np.save(os.path.join(base_folder, "X_train.npy"), X_train)
    np.save(os.path.join(base_folder, "X_test.npy"), X_test)
    np.save(os.path.join(base_folder, "y_train.npy"), y_train)
    np.save(os.path.join(base_folder, "y_test.npy"), y_test)


def process_horizon(filepath, folder_name, sequence_length, scaler_filename):
    """
    Run the full sequence generation pipeline for a single time horizon.
    """
    print(f"=== Processing {folder_name.capitalize()} Data ===")
    
    # 1. Load Dataset
    dataset = load_dataset(filepath)
    if dataset is None:
        return
        
    # 2. Extract Multivariate Features
    try:
        feature_data = prepare_features(dataset)
    except ValueError as e:
        print(f"Error: {e}")
        return
        
    # 3. Normalize Data (Fit and transform, saving the scaler)
    scaled_data, _ = scale_features(feature_data, scaler_filename, fit=True)
    
    # 4. Create Multivariate Sequences
    X, y = create_sequences(scaled_data, sequence_length)
    
    # 5. Split Dataset
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # 6. Save Sequences
    save_sequences(X_train, X_test, y_train, y_test, folder_name)
    print("Sequence Generation Completed Successfully.\n" + "-" * 50 + "\n")


def main():
    """
    Run the sequence generation pipeline for all four datasets.
    """
    # Hourly
    process_horizon(
        filepath=os.path.join("data", "hourly.csv"),
        folder_name="hour",
        sequence_length=24,
        scaler_filename="scaler_hourly.pkl"
    )
    # Daily
    process_horizon(
        filepath=os.path.join("data", "daily.csv"),
        folder_name="day",
        sequence_length=30,
        scaler_filename="scaler_daily.pkl"
    )
    # Monthly
    process_horizon(
        filepath=os.path.join("data", "monthly.csv"),
        folder_name="month",
        sequence_length=12,
        scaler_filename="scaler_monthly.pkl"
    )
    # Weekly
    process_horizon(
        filepath=os.path.join("data", "weekly.csv"),
        folder_name="week",
        sequence_length=4,
        scaler_filename="scaler_weekly.pkl"
    )

if __name__ == "__main__":
    main()
