# =============================================================================
# Project: Multi-Horizon Power Consumption Forecasting using Deep Learning
# File: src/model_training.py
# Description: Trains the multivariate LSTM neural networks for all horizons.
# =============================================================================

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def load_sequence_data(folder_name):
    print("Loading Sequence Data...")
    data_dir = os.path.join("data", "processed", folder_name)
    
    if not os.path.exists(data_dir):
        print(f"Sequence files not found for {folder_name}.")
        return None, None, None, None
        
    try:
        X_train = np.load(os.path.join(data_dir, "X_train.npy"))
        X_test = np.load(os.path.join(data_dir, "X_test.npy"))
        y_train = np.load(os.path.join(data_dir, "y_train.npy"))
        y_test = np.load(os.path.join(data_dir, "y_test.npy"))
    except FileNotFoundError:
        print("Sequence files not found.")
        return None, None, None, None
        
    print(f"Dataset Name: {folder_name.capitalize()} Data")
    print(f"Training Shape (X): {X_train.shape}")
    print(f"Testing Shape (X): {X_test.shape}\n")
    return X_train, X_test, y_train, y_test

def build_lstm_model(sequence_length, number_of_features):
    print("Building LSTM Model...")
    model = Sequential()
    
    # We use None for sequence length to allow flexible prediction lengths (e.g., 1 for UI)
    input_shape = (None, number_of_features)
    
    model.add(LSTM(64, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))
    
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])
    print("Model Built Successfully.\n")
    return model

def train_model(model, X_train, y_train, epochs):
    print(f"Training Started... ({epochs} epochs)")
    
    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )

    print("Training Completed.\n")
    return history

def evaluate_model(model, X_test, y_test):
    print("Evaluating Model...")
    predictions = model.predict(X_test, verbose=0)
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)
    
    print("--- Model Performance on Test Data ---")
    print(f"Loss (MSE) : {mse:.4f}")
    print(f"MAE        : {mae:.4f}")
    print(f"RMSE       : {rmse:.4f}")
    print(f"R2 Score   : {r2:.4f}")
    print("--------------------------------------\n")

def save_model(model, dataset_type):
    print("Saving Model...")
    save_folder = "models"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        
    filename = f"{dataset_type}_model.keras"
    save_path = os.path.join(save_folder, filename)
    model.save(save_path)
    print(f"Model saved to {save_path}\n")

def process_training(dataset_type):
    print(f"=== Starting Training Pipeline for: {dataset_type.upper()} Data ===\n")
    X_train, X_test, y_train, y_test = load_sequence_data(dataset_type)
    
    if X_train is None:
        return
        
    # Dynamic feature extraction
    sequence_length = X_train.shape[1]
    number_of_features = X_train.shape[2]
    
    model = build_lstm_model(sequence_length, number_of_features)
    # Set epochs based on forecasting horizon
    epoch_map = {
        "hour": 15,
        "day": 50,
        "week": 75,
        "month": 100
    }

    epochs = epoch_map.get(dataset_type, 25)

    history = train_model(model, X_train, y_train, epochs)
    evaluate_model(model, X_test, y_test)
    save_model(model, dataset_type)
    print("Training Finished Successfully.\n" + "="*50 + "\n")

def main():
    # Train all 4 horizons sequentially
    horizons = ["hour", "day", "week", "month"]
    for h in horizons:
        process_training(h)

if __name__ == "__main__":
    main()
