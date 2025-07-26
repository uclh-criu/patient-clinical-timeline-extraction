import os
import pandas as pd
import json
import sys
import importlib

# Import our scripts as modules
import extract_relative_dates
import run_inference
import evaluate_predictions
import generate_patient_timelines

# Import configuration
import config

def main():
    """
    Run the complete extraction, evaluation, and timeline generation pipeline
    by calling each of the specialized scripts in sequence.
    
    This main function serves as a convenience wrapper around the four scripts:
    1. extract_relative_dates.py: Extracts relative dates from clinical notes using LLMs
    2. run_inference.py: Performs extraction and saves predictions to CSV
    3. evaluate_predictions.py: Evaluates predictions against gold standards
    4. generate_patient_timelines.py: Creates patient timeline artifacts
    """
    print(f"=== Running Complete Pipeline for {config.EXTRACTION_METHOD} ===")
    
    # Step 1: Extract relative dates (if enabled)
    print("\n[STEP 1] Extracting relative dates...")
    extract_relative_dates.extract_relative_dates()
    
    # Step 2: Run inference
    print("\n[STEP 2] Running inference...")
    run_inference.run_inference()
    
    # Step 3: Evaluate predictions
    print("\n[STEP 3] Evaluating predictions...")
    evaluate_predictions.evaluate_predictions()
    
    # Step 4: Generate patient timelines
    print("\n[STEP 4] Generating patient timelines...")
    generate_patient_timelines.create_patient_timelines()
    
    print("\n=== Pipeline Completed Successfully ===")

if __name__ == "__main__":
    # Run the complete pipeline
    main()