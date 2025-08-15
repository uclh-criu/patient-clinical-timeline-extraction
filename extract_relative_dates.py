import os
import pandas as pd
import json
import sys
from datetime import datetime
from tqdm import tqdm

# Import from our modules
import config

# Import from utility modules
from utils.inference_eval_utils import get_data_path, safe_json_loads, transform_python_to_json
from utils.relative_date_utils import extract_relative_dates_llm

def extract_relative_dates():
    """
    Extract relative dates from clinical notes using LLMs and save to CSV.
    
    This script:
    1. Loads the dataset specified in config
    2. Checks if relative date extraction is enabled
    3. Processes each note to extract relative dates using an LLM
    4. Saves the extracted dates back to the CSV in a new column
    
    Requirements:
    - config.ENABLE_RELATIVE_DATE_EXTRACTION must be True
    - config.TIMESTAMP_COLUMN must be defined and present in the dataset
    """
    print(f"\n=== Extracting Relative Dates ===")
    
    # Check if relative date extraction is enabled
    if not hasattr(config, 'ENABLE_RELATIVE_DATE_EXTRACTION') or not config.ENABLE_RELATIVE_DATE_EXTRACTION:
        print("Relative date extraction is disabled in config.py. Set ENABLE_RELATIVE_DATE_EXTRACTION = True to enable.")
        return
        
    # Get dataset path from config
    dataset_path = get_data_path(config)
    if not dataset_path:
        print(f"Error: Could not determine dataset path from config")
        return
        
    # Check if timestamp column is configured
    if not hasattr(config, 'TIMESTAMP_COLUMN') or not config.TIMESTAMP_COLUMN:
        print(f"Error: TIMESTAMP_COLUMN not defined in config.py")
        return
        
    timestamp_column = config.TIMESTAMP_COLUMN
    text_column = config.TEXT_COLUMN
    
    # Print configuration
    print(f"Dataset path: {dataset_path}")
    print(f"Text column: {text_column}")
    print(f"Timestamp column: {timestamp_column}")
    print(f"LLM model: {config.RELATIVE_DATE_LLM_MODEL}")
    
    # Load dataset
    try:
        df = pd.read_csv(dataset_path)
        print(f"Loaded dataset with {len(df)} rows")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
        
    # Check if required columns exist
    if text_column not in df.columns:
        print(f"Error: Text column '{text_column}' not found in dataset")
        return
        
    if timestamp_column not in df.columns:
        print(f"Error: Timestamp column '{timestamp_column}' not found in dataset")
        return
        
    # Add column for extracted dates if it doesn't exist
    if 'llm_extracted_dates' not in df.columns:
        df['llm_extracted_dates'] = None
        
    # Process each note
    notes_processed = 0
    dates_extracted = 0
    
    # Process each note
    with tqdm(total=len(df), desc="Extracting relative dates", unit="note") as pbar:
        for i, row in df.iterrows():
            # Get text and timestamp
            text = str(row.get(text_column, ''))
            timestamp_str = row.get(timestamp_column)
            
            # Skip if no text or timestamp
            if not text or pd.isna(text) or not timestamp_str or pd.isna(timestamp_str):
                pbar.update(1)
                continue
                
            # Parse timestamp
            document_timestamp = None
            timestamp_formats = [
                '%Y-%m-%d',              # 2023-10-26
                '%Y-%m-%d %H:%M:%S',     # 2023-10-26 15:30:45
                '%m/%d/%Y',              # 10/26/2023
                '%m/%d/%Y %H:%M:%S',     # 10/26/2023 15:30:45
                '%d-%b-%Y',              # 26-Oct-2023
                '%d %b %Y',              # 26 Oct 2023
                '%d/%m/%Y',              # 14/05/2025
            ]
            
            for format_str in timestamp_formats:
                try:
                    document_timestamp = datetime.strptime(str(timestamp_str), format_str)
                    break
                except ValueError:
                    continue
                    
            if not document_timestamp:
                print(f"Warning: Could not parse timestamp '{timestamp_str}' for row {i}")
                pbar.update(1)
                continue
                
            # Extract relative dates
            try:
                relative_dates = extract_relative_dates_llm(text, document_timestamp, config)
                
                if relative_dates:
                    # Convert relative dates to JSON for storage
                    relative_dates_json = []
                    for date_tuple in relative_dates:
                        parsed_date, original_phrase, start_pos = date_tuple
                        relative_dates_json.append({
                            "parsed": parsed_date,
                            "original": original_phrase,
                            "start": start_pos
                        })
                    
                    # Store in the dataframe
                    df.at[i, 'llm_extracted_dates'] = json.dumps(relative_dates_json)
                    
                    # Update counters
                    dates_extracted += len(relative_dates)
                    
                    # Save every 10 rows to avoid data loss if process is interrupted
                    if i % 10 == 0 and i > 0:
                        df.to_csv(dataset_path, index=False)
                
                notes_processed += 1
                
            except Exception as e:
                print(f"Error extracting relative dates for row {i}: {e}")
                
            pbar.update(1)
    
    # Save final results
    try:
        df.to_csv(dataset_path, index=False)
        print(f"Saved updated CSV with relative dates to {dataset_path}")
    except Exception as e:
        print(f"Error saving updated CSV: {e}")
        
    # Print summary
    print("\n=== Relative Date Extraction Summary ===")
    print(f"Total notes processed: {notes_processed}")
    print(f"Total relative dates extracted: {dates_extracted}")
    print(f"Average dates per note: {dates_extracted/max(1, notes_processed):.2f}")
    print(f"Results saved to column 'llm_extracted_dates' in {dataset_path}")
    print("=======================================")

if __name__ == "__main__":
    # Run the relative date extraction
    extract_relative_dates()
