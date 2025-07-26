import os
import pandas as pd
import json
import sys
import importlib

# Import our scripts as modules
import run_inference
import evaluate_predictions
import generate_patient_timelines

# Import configuration
import config

def main():
    """
    Run the complete extraction, evaluation, and timeline generation pipeline
    by calling each of the specialized scripts in sequence.
    
    This main function serves as a convenience wrapper around the three scripts:
    1. run_inference.py: Performs extraction and saves predictions to CSV
    2. evaluate_predictions.py: Evaluates predictions against gold standards
    3. generate_patient_timelines.py: Creates patient timeline artifacts
    """
    print(f"=== Running Complete Pipeline for {config.EXTRACTION_METHOD} ===")
    
    # Step 1: Run inference
    print("\n[STEP 1] Running inference...")
    run_inference.run_inference()
    
    # Step 2: Evaluate predictions
    print("\n[STEP 2] Evaluating predictions...")
    evaluate_predictions.evaluate_predictions()
    
    # Step 3: Generate patient timelines
    print("\n[STEP 3] Generating patient timelines...")
    generate_patient_timelines.create_patient_timelines()
    
    print("\n=== Pipeline Completed Successfully ===")

if __name__ == "__main__":
    # Run the complete pipeline
    main()