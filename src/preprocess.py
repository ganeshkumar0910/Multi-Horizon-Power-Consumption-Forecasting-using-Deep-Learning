# =============================================================================
# Project: Multi-Horizon Power Consumption Forecasting using Deep Learning
# File: src/preprocess.py
# Description: This script cleans and prepares the dataset for our Deep Learning models.
#              It handles loading, cleaning, formatting, and saving the data.
# =============================================================================

# We import 'os' to help us check if files and folders exist on the computer.
import os

# We import 'pandas' because it is the standard library in Python for handling data.
# 'pd' is a common short name for pandas to save typing.
import pandas as pd


def load_dataset(filepath):
    """
    Purpose: Load the PJME_hourly.csv dataset from the given filepath.
    It prints out basic details like the first few rows, shape, column names, and data types.
    """
    print("--- Loading Dataset... ---")
    
    # Check if the file exists before trying to open it
    if not os.path.exists(filepath):
        print("Dataset not found.")
        print("Please place PJME_hourly.csv inside Dataset folder.")
        # Return None to indicate we failed to load the data
        return None
        
    # Read the CSV file using pandas
    dataset = pd.read_csv(filepath)
    
    print("Dataset Loaded Successfully")
    print()
    
    # Display the first five rows to see what the data looks like
    print("First five rows:")
    print(dataset.head())
    print()
    
    # Display the shape (number of rows and columns)
    print("Shape of the dataset (rows, columns):")
    print(dataset.shape)
    print()
    
    # Display the column names
    print("Column names:")
    print(dataset.columns.tolist())
    print()
    
    # Display the data types (e.g., text, numbers, dates)
    print("Data types:")
    print(dataset.dtypes)
    print()
    
    return dataset


def check_missing_values(dataset):
    """
    Purpose: Check for missing (empty) values in the dataset.
    Missing values can cause problems for Deep Learning models, so we must find them.
    """
    print("--- Checking Missing Values... ---")
    
    # dataset.isnull() checks every single cell for a missing value (returns True/False)
    # .sum() adds them up for each column
    missing_values = dataset.isnull().sum()
    print("Missing values in each column:")
    print(missing_values)
    print()
    
    # Calculate the total number of missing values across the whole dataset
    total_missing = missing_values.sum()
    print("Total missing values in the entire dataset:", total_missing)
    
    # Calculate the percentage of missing values
    # Total cells = number of rows * number of columns
    total_cells = dataset.shape[0] * dataset.shape[1]
    percentage_missing = (total_missing / total_cells) * 100
    
    # The round() function makes the number easier to read (2 decimal places)
    print("Percentage of missing values:", round(percentage_missing, 2), "%")
    print()
    
    return dataset


def remove_duplicates(dataset):
    """
    Purpose: Find and remove duplicate rows in the dataset.
    Duplicate data doesn't provide new information and can slow down training.
    """
    print("--- Removing Duplicates... ---")
    
    # Find out how many duplicate rows exist
    number_of_duplicates = dataset.duplicated().sum()
    print("Number of duplicate rows found:", number_of_duplicates)
    
    # If there are duplicates, we remove them using drop_duplicates()
    if number_of_duplicates > 0:
        dataset = dataset.drop_duplicates()
        print("Duplicates removed.")
    else:
        print("No duplicates found to remove.")
        
    # Print the new size of the dataset
    print("Updated dataset size (rows, columns):", dataset.shape)
    print()
    
    return dataset


def convert_datetime(dataset):
    """
    Purpose: Convert the 'Datetime' column from standard text (string) into a datetime datatype.
    A datetime datatype allows us to easily sort by time and extract things like hour, day, or year.
    """
    print("--- Converting Datetime... ---")
    
    # We use a try-except block to safely handle potential errors during conversion
    try:
        # pd.to_datetime() changes the column from text to datetime objects
        # 'errors="coerce"' will put 'NaT' (Not a Time) if it finds an invalid date instead of crashing
        dataset['Datetime'] = pd.to_datetime(dataset['Datetime'], errors='coerce')
        print("Datetime conversion successful.")
    except Exception as error_message:
        print("Error during Datetime conversion:", error_message)
        
    print()
    
    return dataset


def sort_dataset(dataset):
    """
    Purpose: Sort the dataset based on the 'Datetime' column.
    For Time-Series forecasting, the data must be in chronological (time) order.
    """
    print("--- Sorting Dataset... ---")
    
    # sort_values() sorts the rows based on a specific column
    dataset = dataset.sort_values(by='Datetime')
    
    print("Dataset sorted by Datetime.")
    print()
    
    return dataset


def set_datetime_index(dataset):
    """
    Purpose: Set the 'Datetime' column as the index (row labels) of the DataFrame.
    In pandas, time-series data is much easier to work with if the time is the index.
    """
    print("--- Setting Datetime Index... ---")
    
    # set_index() makes the chosen column the row identifier
    dataset = dataset.set_index('Datetime')
    
    print("Datetime is now set as the index.")
    print()
    
    return dataset


def check_dataset_info(dataset):
    """
    Purpose: Display final information about the dataset after all cleaning steps.
    """
    print("--- Displaying Dataset Info... ---")
    
    print("Final Shape:")
    print(dataset.shape)
    print()
    
    # .info() gives a summary of the DataFrame, including memory usage
    print("Information and Memory Usage:")
    dataset.info()
    print()
    
    # .describe() gives statistical summaries like mean, min, max for numerical columns
    print("Statistical Summary:")
    print(dataset.describe())
    print()
    
    return dataset


def save_clean_dataset(dataset):
    """
    Purpose: Save the cleaned dataset to a new CSV file.
    This way, we don't have to clean the data every time we run the project.
    """
    print("--- Saving Dataset... ---")
    
    # Define where we want to save the file
    save_folder = os.path.join("data", "cleaned")
    save_filename = "cleaned_power_data.csv"
    
    # os.path.join helps us create a safe path like 'data/cleaned/cleaned_power_data.csv'
    save_path = os.path.join(save_folder, save_filename)
    
    # Make sure the folder exists. If not, create it.
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        print("Created directory:", save_folder)
        
    # Save the dataframe to a CSV file without saving the index if we didn't want it, 
    # but here Datetime is the index, so we must save the index!
    dataset.to_csv(save_path, index=True)
    
    print("Clean dataset saved to:", save_path)
    print()


def preprocess_pipeline():
    """
    Purpose: Run all the preprocessing functions in the correct order.
    This acts as the master function that controls the flow of the program.
    """
    # The path to the original dataset
    filepath = os.path.join("Dataset", "PJMW_hourly.csv")
    
    # Step 1: Load Dataset
    dataset = load_dataset(filepath)
    
    # If the dataset didn't load (maybe it was missing), we stop the pipeline
    if dataset is None:
        return
        
    # Step 2: Check Missing Values
    dataset = check_missing_values(dataset)
    
    # Step 3: Remove Duplicates
    dataset = remove_duplicates(dataset)
    
    # Step 4: Convert Datetime
    dataset = convert_datetime(dataset)
    
    # Step 5: Sort Dataset
    dataset = sort_dataset(dataset)
    
    # Step 6: Set Datetime Index
    dataset = set_datetime_index(dataset)
    
    # Step 7: Display Dataset Info
    dataset = check_dataset_info(dataset)
    
    # Step 8: Save Clean Dataset
    save_clean_dataset(dataset)
    
    # Final success message
    print("Preprocessing Completed Successfully")


def main():
    """
    Purpose: The entry point of this script. It starts the preprocess_pipeline.
    """
    preprocess_pipeline()


# This tells Python to run the main() function if this file is run directly
if __name__ == "__main__":
    main()
