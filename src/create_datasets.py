# =============================================================================
# Project: Multi-Horizon Power Consumption Forecasting using Deep Learning
# File: src/create_datasets.py
# Description: This script creates four different datasets (hourly, daily, 
#              monthly, weekly) from our main feature engineered dataset.
#              These datasets will allow us to train different models for 
#              different forecasting horizons.
# =============================================================================

# os helps us check if files and folders exist on the computer
import os

# pandas is used to manipulate data, particularly resampling for time series
import pandas as pd


def load_dataset(filepath):
    """
    Purpose: Load the feature engineered dataset.
    """
    print("Loading Dataset...")
    
    # Check if the dataset exists
    if not os.path.exists(filepath):
        print("Feature engineered dataset not found.")
        print("Run feature_engineering.py first.")
        # Return None to stop without crashing
        return None
        
    # Read the CSV file. We set Datetime as the index because we are working with time series.
    dataset = pd.read_csv(filepath, index_col='Datetime', parse_dates=True)
    
    print("Dataset Loaded Successfully")
    print()
    
    print("First Five Rows:")
    print(dataset.head())
    print()
    
    print("Dataset Shape (rows, columns):")
    print(dataset.shape)
    print()
    
    return dataset


def create_hourly_dataset(dataset):
    """
    Purpose: Create the hourly dataset.
    
    Why we create four datasets:
    Different business goals require different forecasting horizons. A power grid 
    manager might need to know the next hour's demand, while a city planner might 
    need to know next week's demand.
    
    Why hourly dataset is used for next hour prediction:
    The hourly dataset contains the highest level of detail. To predict what will 
    happen in the very next hour, we need to look at hour-by-hour patterns.
    """
    print("Creating Hourly Dataset...")
    
    # We use the original dataset directly because it is already hourly.
    # No resampling or aggregation is needed.
    hourly_dataset = dataset.copy()
    
    return hourly_dataset


def create_daily_dataset(dataset):
    """
    Purpose: Create the daily dataset by aggregating the hourly data.
    
    What resample() does:
    Resample is like a 'groupby' function but specifically for dates and times.
    It groups all the hours in a single day together.
    
    Why aggregation is necessary:
    We must aggregate (combine) 24 separate hourly readings into 1 single daily reading.
    We use the mean (average) to represent the typical power consumption for that day.
    
    Why daily average is useful:
    It helps us predict the next day's overall power demand by removing hourly noise.
    """
    print("Creating Daily Dataset...")
    
    # 'D' stands for Daily frequency. We calculate the mean for each day.
    # Select only numeric columns
    numeric_dataset = dataset.select_dtypes(include=["number"])

    # Calculate daily averages
    daily_dataset = numeric_dataset.resample("D").mean()
    
    print("Daily Dataset Shape:")
    print(daily_dataset.shape)
    print()
    
    print("First Five Rows (Daily):")
    print(daily_dataset.head())
    print()
    
    return daily_dataset


def create_monthly_dataset(dataset):
    """
    Purpose: Create the monthly dataset by aggregating the data.
    
    Why monthly forecasting needs monthly data:
    To predict power usage for an entire month, looking at hourly data is too chaotic. 
    By resampling to monthly data, the model can easily learn long-term seasonal 
    trends (like high usage in summer vs winter).
    """
    print("Creating Monthly Dataset...")
    
    # 'ME' stands for Month End frequency. 
    # (Note: The older 'M' is being deprecated in newer pandas versions, so we use 'ME' safely).
    numeric_dataset = dataset.select_dtypes(include=["number"])

    monthly_dataset = numeric_dataset.resample("ME").mean()
    
    print("Monthly Dataset Shape:")
    print(monthly_dataset.shape)
    print()
    
    return monthly_dataset


def create_weekly_dataset(dataset):
    """
    Purpose: Create the weekly dataset by aggregating the data.
    
    Why weekly forecasting needs weekly data:
    Weekly data smooths out all daily changes and noise, allowing the model 
    to focus purely on week-by-week trends.
    """
    print("Creating Weekly Dataset...")
    
    # 'W' stands for Weekly frequency.
    numeric_dataset = dataset.select_dtypes(include=["number"])

    weekly_dataset = numeric_dataset.resample("W").mean()
    
    print("Weekly Dataset Shape:")
    print(weekly_dataset.shape)
    print()
    
    return weekly_dataset


def display_dataset_summary(datasets_dict):
    """
    Purpose: Display a clean summary of all created datasets.
    """
    print("--- Dataset Summary ---")
    
    # We loop through our dictionary of datasets to print their details
    for name, data in datasets_dict.items():
        print(f"Dataset Name: {name}.csv")
        print(f"Rows: {data.shape[0]}")
        print(f"Columns: {data.shape[1]}")
        print(f"Start Date: {data.index.min()}")
        print(f"End Date: {data.index.max()}")
        print(f"Missing Values: {data.isnull().sum().sum()}")
        print("-" * 25)
        
    print()


def save_datasets(datasets_dict):
    """
    Purpose: Save the four datasets as CSV files inside the data/ folder.
    """
    print("Saving Datasets...")
    
    # Define the output directory path
    save_folder = "data"
    
    # Create the directory if it does not exist
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        
    # Loop through the dictionary and save each dataset
    for name, data in datasets_dict.items():
        # Create the filename by adding .csv
        filename = f"{name}.csv"
        save_path = os.path.join(save_folder, filename)
        
        # Save to CSV, keeping the Datetime index
        data.to_csv(save_path, index=True)
        
    print("Datasets Created Successfully.")


def main():
    """
    Purpose: Execute all functions in order to create the datasets.
    """
    # The path to the feature engineered dataset
    filepath = os.path.join("data", "processed", "feature_engineered_data.csv")
    
    # Step 1: Load Dataset
    dataset = load_dataset(filepath)
    
    # Stop if dataset was not found
    if dataset is None:
        return
        
    # Step 2: Create Hourly Dataset
    hourly = create_hourly_dataset(dataset)
    
    # Step 3: Create Daily Dataset
    daily = create_daily_dataset(dataset)
    
    # Step 4: Create Monthly Dataset
    monthly = create_monthly_dataset(dataset)
    
    # Step 5: Create Weekly Dataset
    weekly = create_weekly_dataset(dataset)
    
    # Put datasets in a dictionary for easy summarizing and saving
    datasets_dict = {
        "hourly": hourly,
        "daily": daily,
        "monthly": monthly,
        "weekly": weekly
    }
    
    # Step 6: Display Summary
    display_dataset_summary(datasets_dict)
    
    # Step 7: Save All Datasets
    save_datasets(datasets_dict)


# This checks if the file is being run directly by the user
if __name__ == "__main__":
    main()
