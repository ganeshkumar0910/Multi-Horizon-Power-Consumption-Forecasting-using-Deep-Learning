import os
import pandas as pd
from src.utils import FEATURE_COLUMNS

def generate_sample(dataset_filename, output_filename, sequence_length):
    dataset_path = os.path.join("data", dataset_filename)
    if not os.path.exists(dataset_path):
        print(f"Dataset {dataset_path} not found.")
        return
        
    df = pd.read_csv(dataset_path)
    
    # Extract only the exact feature columns in order
    sample_df = df[FEATURE_COLUMNS].copy()
    
    # Get the last N rows (sequence_length)
    sample_df = sample_df.tail(sequence_length)
    
    # Save the sample
    os.makedirs("samples", exist_ok=True)
    output_path = os.path.join("samples", output_filename)
    sample_df.to_csv(output_path, index=False)
    print(f"Generated {output_path} with shape {sample_df.shape}")

if __name__ == "__main__":
    generate_sample("hourly.csv", "sample_hour.csv", 24)
    generate_sample("daily.csv", "sample_day.csv", 30)
    generate_sample("weekly.csv", "sample_week.csv", 4)
    generate_sample("monthly.csv", "sample_month.csv", 12)
