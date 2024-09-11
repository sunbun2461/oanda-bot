import os
import pandas as pd

# Define directories
input_dir = 'historical_data/'
output_dir = 'cleaned_data/'

# Ensure output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Function to clean and save data
def clean_data(filename):
    df = pd.read_csv(filename)
    
    # Step 1: Remove duplicates
    df_cleaned = df.drop_duplicates(subset='time')
    
    # Step 2: Handle missing values (drop rows with any missing data)
    df_cleaned = df_cleaned.dropna()
    
    # Step 3: Convert the time column to datetime format
    df_cleaned['time'] = pd.to_datetime(df_cleaned['time'])
    
    # Step 4: Check for gaps in time (assuming 5 minutes interval)
    df_cleaned = df_cleaned.sort_values(by='time')
    time_diff = df_cleaned['time'].diff().dt.total_seconds().dropna()
    gaps = time_diff[time_diff > 300]  # 300 seconds = 5 minutes
    if not gaps.empty:
        print(f"Gaps found in {filename}: \n{gaps}")
    
    # Save the cleaned data to the output directory
    output_filename = os.path.join(output_dir, os.path.basename(filename))
    df_cleaned.to_csv(output_filename, index=False)
    print(f"Data cleaned and saved to {output_filename}")

# Iterate over all files in the historical_data folder
for file in os.listdir(input_dir):
    if file.endswith(".csv"):
        filepath = os.path.join(input_dir, file)
        clean_data(filepath)

print("All files processed and cleaned.")