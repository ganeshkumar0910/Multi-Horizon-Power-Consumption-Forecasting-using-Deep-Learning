# =============================================================================
# Project: Multi-Horizon Power Consumption Forecasting using Deep Learning
# File: src/eda.py
# Description: This script performs Exploratory Data Analysis (EDA) on the 
#              cleaned dataset. It generates various plots to help us understand 
#              the data patterns before we build any Deep Learning models.
# =============================================================================

# os is used to check for file existence and create folders
import os

# pandas is used to load and manipulate the dataset
import pandas as pd

# matplotlib.pyplot is used to draw graphs and charts
import matplotlib.pyplot as plt

# seaborn is used to make our graphs look more beautiful and professional
import seaborn as sns


def load_dataset(filepath):
    """
    Purpose: Load the cleaned dataset from the given filepath.
    It prints out the first 5 rows, last 5 rows, and the shape.
    """
    print("Loading Dataset...")
    
    # Check if the cleaned file exists
    if not os.path.exists(filepath):
        print("Cleaned dataset not found.")
        print("Run preprocess.py first.")
        # Return None to safely stop without crashing
        return None
        
    # Read the CSV file. 
    # index_col='Datetime' tells pandas to use the 'Datetime' column as the row index.
    # parse_dates=True automatically converts the index to a datetime format.
    dataset = pd.read_csv(filepath, index_col='Datetime', parse_dates=True)
    
    print("Dataset Loaded Successfully")
    print()
    
    # Display the first 5 rows
    print("First 5 rows:")
    print(dataset.head())
    print()
    
    # Display the last 5 rows
    print("Last 5 rows:")
    print(dataset.tail())
    print()
    
    # Display the dataset shape
    print("Dataset Shape (rows, columns):")
    print(dataset.shape)
    print()
    
    return dataset


def dataset_information(dataset):
    """
    Purpose: Display basic technical information about the dataset.
    """
    print("Generating Dataset Information...")
    print()
    
    print("Column names:")
    print(dataset.columns.tolist())
    print()
    
    print("Data types:")
    print(dataset.dtypes)
    print()
    
    # The .info() method prints memory usage and other details directly to the console
    print("Memory usage and structure:")
    dataset.info()
    print()
    
    print("Summary statistics:")
    print(dataset.describe())
    print()


def check_missing_values(dataset):
    """
    Purpose: Display missing values in the console and create a bar chart.
    """
    print("Checking Missing Values...")
    
    # Calculate missing values
    missing_values = dataset.isnull().sum()
    print("Missing values per column:")
    print(missing_values)
    print()
    
    # Create a bar chart for missing values
    fig = plt.figure(figsize=(10, 6))
    missing_values.plot(kind='bar', color='orange')
    
    # Add title and labels
    plt.title('Missing Values in Dataset')
    plt.xlabel('Columns')
    plt.ylabel('Number of Missing Values')
    plt.grid(True)
    
    # We return the figure so it can be saved later
    return fig


def plot_hourly_consumption(dataset):
    """
    Purpose: Create a line chart showing the hourly electricity consumption over time.
    """
    print("Creating Hourly Plot...")
    
    fig = plt.figure(figsize=(15, 6))
    
    # Plot Datetime (which is the index) vs PJME_MW
    plt.plot(dataset.index, dataset['PJMW_MW'], color='blue', linewidth=0.5)
    
    # Add labels and title
    plt.title('Hourly Electricity Consumption')
    plt.xlabel('Datetime')
    plt.ylabel('Power Consumption (MW)')
    
    # Add a grid for easier reading
    plt.grid(True)
    
    return fig


def plot_daily_average(dataset):
    """
    Purpose: Resample the data to daily frequency using the mean (average), and plot it.
    """
    print("Creating Daily Plot...")
    
    # Resample 'D' means Daily. .mean() calculates the daily average.
    daily_data = dataset.resample('D').mean()
    
    fig = plt.figure(figsize=(15, 6))
    plt.plot(daily_data.index, daily_data['PJMW_MW'], color='green', linewidth=1)
    
    plt.title('Daily Average Electricity Consumption')
    plt.xlabel('Date')
    plt.ylabel('Average Power Consumption (MW)')
    plt.grid(True)
    
    return fig


def plot_monthly_average(dataset):
    """
    Purpose: Resample the data to monthly frequency using the mean (average), and plot it.
    """
    print("Creating Monthly Plot...")
    
    # Resample 'ME' means Monthly End. .mean() calculates the monthly average.
    monthly_data = dataset.resample('ME').mean()
    
    fig = plt.figure(figsize=(15, 6))
    plt.plot(monthly_data.index, monthly_data['PJMW_MW'], color='red', marker='o')
    
    plt.title('Monthly Average Electricity Consumption')
    plt.xlabel('Month')
    plt.ylabel('Average Power Consumption (MW)')
    plt.grid(True)
    
    return fig


def plot_weekly_average(dataset):
    """
    Purpose: Resample the data to weekly frequency using the mean (average), and plot it.
    """
    print("Creating Weekly Plot...")
    
    # Resample 'W' means Weekly.
    weekly_data = dataset.resample('W').mean()
    
    fig = plt.figure(figsize=(15, 5))
    plt.plot(weekly_data.index, weekly_data['PJMW_MW'], color='purple', marker='s', linewidth=2)
    plt.grid(True)
    plt.title('Weekly Average Electricity Consumption')
    plt.xlabel('Week')
    plt.ylabel('Average Power Consumption (MW)')
    plt.grid(True)
    
    return fig


def plot_distribution(dataset):
    """
    Purpose: Create a Histogram and Kernel Density Estimate (KDE) to see the distribution of data.
    """
    print("Creating Distribution Plot...")
    
    fig = plt.figure(figsize=(12, 6))
    
    # sns.histplot creates both the histogram bars and the smooth KDE line
    sns.histplot(dataset['PJMW_MW'], kde=True, color='teal', bins=50)
    
    plt.title('Distribution of Power Consumption')
    plt.xlabel('Power Consumption (MW)')
    plt.ylabel('Frequency')
    plt.grid(True)
    
    return fig


def plot_boxplot(dataset):
    """
    Purpose: Create a boxplot to visually detect any outliers in the data.
    """
    print("Creating Box Plot...")
    
    fig = plt.figure(figsize=(10, 6))
    
    # sns.boxplot draws the boxplot
    sns.boxplot(x=dataset['PJMW_MW'], color='lightblue')
    
    plt.title('Boxplot of Power Consumption (Outlier Detection)')
    plt.xlabel('Power Consumption (MW)')
    # Y-axis label added for completeness, even if boxplot is horizontal
    plt.ylabel('PJME Region') 
    plt.grid(True)
    
    return fig


def plot_correlation_heatmap(dataset):
    """
    Purpose: Generate a correlation matrix and plot it as a heatmap.
    """
    print("Creating Correlation Heatmap...")
    
    # Calculate the correlation matrix
    correlation_matrix = dataset.corr()
    
    fig = plt.figure(figsize=(8, 6))
    
    # sns.heatmap visualizes the correlation matrix with colors
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    
    plt.title('Correlation Heatmap')
    plt.xlabel('Features')
    plt.ylabel('Features')
    
    return fig


def plot_rolling_mean(dataset):
    """
    Purpose: Calculate a 24-hour rolling mean (moving average) and plot it alongside the original data.
    """
    print("Creating Rolling Mean Plot...")
    
    # Calculate a 24-hour rolling mean. We use window=24 because our data is hourly.
    rolling_mean_24h = dataset['PJMW_MW'].rolling(window=24).mean()
    
    fig = plt.figure(figsize=(15, 6))
    
    # We plot a small subset (e.g., first 500 hours) so the graph is readable
    subset = dataset.head(500)
    subset_rolling = rolling_mean_24h.head(500)
    
    # Plot both lines on the same graph
    plt.plot(subset.index, subset['PJMW_MW'], color='lightgray', label='Original Data (Hourly)')
    plt.plot(subset_rolling.index, subset_rolling, color='red', label='24-Hour Rolling Mean')
    
    plt.title('Original Data vs 24-Hour Rolling Mean (First 500 Hours)')
    plt.xlabel('Datetime')
    plt.ylabel('Power Consumption (MW)')
    
    # Display the legend to tell which line is which
    plt.legend()
    plt.grid(True)
    
    return fig


def save_plots(figures_dict):
    """
    Purpose: Save every generated graph inside the outputs/plots/ folder.
    """
    print("Saving Plots...")
    
    # Define the output directory path
    save_folder = os.path.join("outputs", "plots")
    
    # Create the directory if it does not exist
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        
    # Loop through our dictionary of figures and save each one
    for filename, fig in figures_dict.items():
        # Combine the folder path and filename
        filepath = os.path.join(save_folder, filename)
        
        # Save the figure to the computer
        fig.savefig(filepath, bbox_inches='tight')
        
        # Close the figure to free up memory
        plt.close(fig)


def main():
    """
    Purpose: Execute all EDA functions in the correct order based on the program flow.
    """
    # The path to the cleaned dataset generated by preprocess.py
    filepath = os.path.join("data", "cleaned", "cleaned_power_data.csv")
    
    # Step 1: Load Dataset
    dataset = load_dataset(filepath)
    
    # If dataset is not found, stop the program
    if dataset is None:
        return
        
    # Step 2: Dataset Information
    dataset_information(dataset)
    
    # Step 3: Missing Values
    fig_missing = check_missing_values(dataset)
    
    # Step 4: Hourly Plot
    fig_hourly = plot_hourly_consumption(dataset)
    
    # Step 5: Daily Plot
    fig_daily = plot_daily_average(dataset)
    
    # Step 6: Monthly Plot
    fig_monthly = plot_monthly_average(dataset)
    
    # Step 7: Weekly Plot
    fig_weekly = plot_weekly_average(dataset)
    
    # Step 8: Distribution Plot
    fig_dist = plot_distribution(dataset)
    
    # Step 9: Box Plot
    fig_box = plot_boxplot(dataset)
    
    # Step 10: Correlation Heatmap
    fig_corr = plot_correlation_heatmap(dataset)
    
    # Step 11: Rolling Mean
    fig_rolling = plot_rolling_mean(dataset)
    
    # Step 12: Save All Plots
    # We collect all figures in a dictionary with their meaningful filenames
    figures_to_save = {
        "missing_values.png": fig_missing,
        "hourly_consumption.png": fig_hourly,
        "daily_consumption.png": fig_daily,
        "monthly_consumption.png": fig_monthly,
        "weekly_consumption.png": fig_weekly,
        "distribution.png": fig_dist,
        "boxplot.png": fig_box,
        "correlation_heatmap.png": fig_corr,
        "rolling_mean.png": fig_rolling
    }
    
    save_plots(figures_to_save)
    
    # Final success message
    print("EDA Completed Successfully.")


# This code block starts the program when the script is run directly
if __name__ == "__main__":
    main()
