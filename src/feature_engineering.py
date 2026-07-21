# =============================================================================
# Project: Multi-Horizon Power Consumption Forecasting using Deep Learning
# File: src/feature_engineering.py
# Description: This script creates new useful features from the raw Datetime 
#              and historical power consumption values to help our Deep Learning
#              models learn patterns better.
# =============================================================================

# os helps us check if files and folders exist on the computer
import os

# pandas is used to load, manipulate, and save the dataset
import pandas as pd


def load_dataset(filepath):
    """
    Purpose: Load the cleaned dataset generated in the preprocessing step.
    """
    print("Loading Dataset...")
    
    # Check if the cleaned file exists
    if not os.path.exists(filepath):
        print("Cleaned dataset not found.")
        print("Run preprocess.py first.")
        # Return None to safely stop the program without crashing
        return None
        
    # Read the CSV file. 
    # index_col='Datetime' uses the Datetime column as the row index.
    # parse_dates=True converts the index to a proper datetime format.
    dataset = pd.read_csv(filepath, index_col='Datetime', parse_dates=True)
    
    print("First five rows:")
    print(dataset.head())
    print()
    
    print("Dataset Shape (rows, columns):")
    print(dataset.shape)
    print()
    
    return dataset


def create_time_features(dataset):
    """
    Purpose: Create new time-based features from the Datetime index.
    
    Why time features are useful: 
    Deep Learning models do not inherently understand "Datetime" strings. 
    By breaking the date down into Hour, Day, Month, etc., the model can 
    learn seasonal patterns (e.g., higher power consumption in summer months 
    or during evening hours).
    """
    print("Creating Time Features...")
    
    # Hour: 0 to 23
    dataset['Hour'] = dataset.index.hour
    
    # Day: 1 to 31
    dataset['Day'] = dataset.index.day
    
    # Month: 1 to 12
    dataset['Month'] = dataset.index.month
    
    # Year, Quarter, Week, and Day_Name were removed per requirements
    
    # Day_of_Week: 0 (Monday) to 6 (Sunday)
    dataset['Day_of_Week'] = dataset.index.dayofweek
    
    # Is_Weekend: True if Day_of_Week is 5 (Saturday) or 6 (Sunday), False otherwise
    # We convert the True/False to 1/0 using .astype(int) for the Deep Learning model
    dataset['Is_Weekend'] = (dataset['Day_of_Week'] >= 5).astype(int)
    
    return dataset


def create_lag_features(dataset):
    """
    Purpose: Create lag features for historical power consumption.
    
    What lag features are:
    Lag features are just the values of the target variable from the past.
    For example, Lag_1 is the power consumption exactly 1 hour ago.
    
    Why lag features are useful:
    They help forecasting because past power consumption is a very strong 
    indicator of future power consumption. If consumption was high 1 hour ago, 
    it will likely be high now.
    """
    print("Creating Lag Features...")
    
    # Lag_1: Power consumption 1 hour ago
    # .shift(1) moves all values down by 1 row
    dataset['Lag_1'] = dataset['PJMW_MW'].shift(1)
    
    # Lag_24: Power consumption 24 hours ago (same time yesterday)
    dataset['Lag_24'] = dataset['PJMW_MW'].shift(24)
    
    # Lag_48: Power consumption 48 hours ago (same time 2 days ago)
    dataset['Lag_48'] = dataset['PJMW_MW'].shift(48)
    
    # Lag_168: Power consumption 168 hours ago (same time exactly 1 week ago)
    dataset['Lag_168'] = dataset['PJMW_MW'].shift(168)
    
    return dataset


def create_rolling_features(dataset):
    """
    Purpose: Create rolling statistics using a 24-hour window.
    
    Why rolling statistics are useful:
    Rolling features capture the recent trend of the data over a specific window of time.
    Instead of just looking at exactly 1 hour ago, rolling features summarize the 
    last 24 hours. This smooths out sudden spikes and gives the model a sense of 
    momentum (e.g., is the overall daily trend going up or down?).
    """
    print("Creating Rolling Features...")
    
    # We create a rolling window of the last 24 hours
    # We use .shift(1) before applying the rolling window to avoid 'data leakage'. 
    # We should only use PAST data to predict the future, not current data.
    past_24_hours = dataset['PJMW_MW'].shift(1).rolling(window=24)
    
    # Rolling_Mean_24: The average power consumption over the last 24 hours
    dataset['Rolling_Mean_24'] = past_24_hours.mean()
    
    # Rolling_Std_24: The standard deviation (how spread out the data is) over the last 24 hours
    dataset['Rolling_Std_24'] = past_24_hours.std()
    
    # Rolling_Min_24: The lowest power consumption over the last 24 hours
    dataset['Rolling_Min_24'] = past_24_hours.min()
    
    # Rolling_Max_24: The highest power consumption over the last 24 hours
    dataset['Rolling_Max_24'] = past_24_hours.max()
    
    return dataset


def check_missing_after_features(dataset):
    """
    Purpose: Display and remove missing values created by lag and rolling features.
    
    Why missing values occur:
    When we shift data down by 168 hours to create Lag_168, the first 168 rows 
    of our dataset won't have any past data to look at, so pandas fills them with 
    NaN (Not a Number).
    
    Why removing NaN values is necessary:
    Deep Learning models require numbers to do math. They will crash if they 
    encounter missing (NaN) values during training.
    """
    print("Checking Missing Values...")
    
    # Count the total number of missing values in the entire dataset
    total_missing = dataset.isnull().sum().sum()
    print("Total missing values after feature engineering:", total_missing)
    print()
    
    print("Removing Missing Values...")
    
    # .dropna() removes any row that contains at least one missing value
    dataset = dataset.dropna()
    
    print("Updated dataset shape after removing NaNs (rows, columns):")
    print(dataset.shape)
    print()
    
    return dataset


def display_feature_summary(dataset):
    """
    Purpose: Print a summary of the newly created feature dataset.
    """
    print("--- Feature Summary ---")
    
    print("Column Names:")
    print(dataset.columns.tolist())
    print()
    
    # Number of features is the number of columns
    num_features = dataset.shape[1]
    print("Number of Features:", num_features)
    print()
    
    print("Dataset Shape:")
    print(dataset.shape)
    print()
    
    print("First Five Rows:")
    print(dataset.head())
    print()


def save_feature_dataset(dataset):
    """
    Purpose: Save the final feature engineered dataset to a CSV file.
    """
    print("Saving Dataset...")
    
    # Define the output directory path
    save_folder = os.path.join("data", "processed")
    save_filename = "feature_engineered_data.csv"
    
    # Combine folder and filename
    save_path = os.path.join(save_folder, save_filename)
    
    # Make sure the directory exists
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        
    # Save the dataframe to a CSV file, keeping the Datetime index
    dataset.to_csv(save_path, index=True)


def main():
    """
    Purpose: Execute every function in the proper order to complete the pipeline.
    """
    # The path to the cleaned dataset generated by preprocess.py
    filepath = os.path.join("data", "cleaned", "cleaned_power_data.csv")
    
    # Step 1: Load Dataset
    dataset = load_dataset(filepath)
    
    # If dataset is missing, stop the pipeline
    if dataset is None:
        return
        
    # Step 2: Create Time Features
    dataset = create_time_features(dataset)
    
    # Step 3: Create Lag Features
    dataset = create_lag_features(dataset)
    
    # Step 4: Create Rolling Features
    dataset = create_rolling_features(dataset)
    
    # Step 5 & 6: Check and Remove Missing Values
    dataset = check_missing_after_features(dataset)
    
    # Step 7: Display Feature Summary
    display_feature_summary(dataset)
    
    # Step 8: Save Dataset
    save_feature_dataset(dataset)
    
    # Final success message
    print("Feature Engineering Completed Successfully.")


# This tells Python to run the main() function if this file is run directly
if __name__ == "__main__":
    main()
