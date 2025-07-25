#!/usr/bin/env python3

import os
import sys
import csv
import glob
import pandas as pd

def replace_paths_in_csv(input_dir):
    """
    Read all _results.csv files in the specified directory, replace path strings,
    and write back to the same files.
    
    Args:
        input_dir (str): Directory containing _results.csv files
    """
    old_path = "/global/scratch/users/jdickerson/2dtm_test_data/"
    new_path = "/data/papers/Leopard-EM_paper_data/"
    
    old_string1 = "_nan_rerun"
    new_string1 = ""

    old_string2 = "temp_nan"
    new_string2 = "all"
    # Find all _results.csv files in the specified directory
    csv_files = glob.glob(os.path.join(input_dir, "**/*_results.csv"), recursive=True)
    
    if not csv_files:
        print(f"No _results.csv files found in {input_dir}")
        return
    
    print(f"Found {len(csv_files)} _results.csv files")
    
    for csv_file in csv_files:
        print(f"Processing: {csv_file}")
        
        try:
            # Read the CSV file using pandas
            df = pd.read_csv(csv_file)
            
            # Replace path strings in all columns
            for col in df.columns:
                if df[col].dtype == 'object':  # Only process string columns
                    df[col] = df[col].astype(str).str.replace(old_path, new_path)
                    df[col] = df[col].astype(str).str.replace(old_string1, new_string1)
                    df[col] = df[col].astype(str).str.replace(old_string2, new_string2)
            
            # Write the modified dataframe back to the file
            df.to_csv(csv_file, index=False)
            print(f"  Successfully updated paths in {csv_file}")
        
        except Exception as e:
            print(f"  Error processing {csv_file}: {e}")
    
    print("Path replacement completed!")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python replace_paths.py <directory_path>")
        sys.exit(1)
    
    input_dir = sys.argv[1]
    
    if not os.path.isdir(input_dir):
        print(f"Error: {input_dir} is not a valid directory")
        sys.exit(1)
    
    replace_paths_in_csv(input_dir)