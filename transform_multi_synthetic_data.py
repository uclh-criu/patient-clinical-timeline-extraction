import pandas as pd
import json
import ast
import re
import random
from copy import deepcopy

def transform_extracted_disorders_to_snomed_umls(disorders_json):
    """
    Transform extracted_disorders into SNOMED and UMLS entity formats.
    
    Args:
        disorders_json (str): JSON string of extracted disorders
    
    Returns:
        tuple: (snomed_entities, umls_entities) as JSON strings
    """
    try:
        if isinstance(disorders_json, str):
            disorders = json.loads(disorders_json)
        else:
            disorders = disorders_json
    except:
        try:
            disorders = ast.literal_eval(disorders_json)
        except:
            return "[]", "[]"
    
    # Create SNOMED entities
    snomed_entities = []
    for disorder in disorders:
        # Copy the base entity
        snomed_entity = {
            "label": disorder["label"],
            "start": disorder["start"],
            "end": disorder["end"],
            "categories": ["disorder"]  # Default category for disorders
        }
        snomed_entities.append(snomed_entity)
    
    # Create UMLS entities (similar to SNOMED but with different categories)
    umls_entities = []
    for disorder in disorders:
        # Map disorder names to appropriate UMLS categories
        category = "Disease or Syndrome"
        if "arthritis" in disorder["label"].lower():
            category = "Disease or Syndrome"
        elif "adenoma" in disorder["label"].lower() or "tumor" in disorder["label"].lower():
            category = "Neoplastic Process"
        elif "pneumonia" in disorder["label"].lower() or "asthma" in disorder["label"].lower():
            category = "Disease or Syndrome"
        elif "headache" in disorder["label"].lower():
            category = "Sign or Symptom"
        
        umls_entity = {
            "label": disorder["label"],
            "start": disorder["start"],
            "end": disorder["end"],
            "categories": [category]
        }
        umls_entities.append(umls_entity)
    
    return json.dumps(snomed_entities), json.dumps(umls_entities)

def transform_relationship_gold(relationship_gold_json):
    """
    Transform relationship_gold from diagnosis_only format to multi_entity format.
    
    Args:
        relationship_gold_json (str): JSON string of relationship gold in diagnosis_only format
    
    Returns:
        str: JSON string of relationship gold in multi_entity format
    """
    try:
        if isinstance(relationship_gold_json, str):
            relationships = json.loads(relationship_gold_json)
        else:
            relationships = relationship_gold_json
    except:
        try:
            relationships = ast.literal_eval(relationship_gold_json)
        except:
            return "[]"
    
    # Transform to multi-entity format
    new_relationships = []
    
    for rel in relationships:
        date = rel.get("date", "")
        diagnoses = rel.get("diagnoses", [])
        
        for diag in diagnoses:
            diagnosis = diag.get("diagnosis", "")
            position = diag.get("position", 0)
            
            # Create new relationship in multi-entity format
            new_rel = {
                "entity_label": diagnosis,
                "entity_category": "diagnosis",  # Default category for disorders
                "date": date
            }
            new_relationships.append(new_rel)
    
    return json.dumps(new_relationships)

def create_empty_prediction_columns():
    """
    Create empty prediction columns with the same structure as in synthetic_updated.csv.
    
    Returns:
        dict: Dictionary with empty prediction columns
    """
    return {
        "naive_(proximity)_predictions": "[]",
        "naive_(proximity)_is_correct": "false",
        "naive_(proximity)_pa_likelihood": "0.0",
        "custom_(pytorch_nn)_predictions": "[]",
        "custom_(pytorch_nn)_is_correct": "false",
        "custom_(pytorch_nn)_pa_likelihood": "0.0",
        "bert_predictions": "[]"
    }

def transform_multi_synthetic_data(input_file, output_file=None):
    """
    Transform synthetic_multi.csv to match the structure of synthetic_updated.csv.
    
    Args:
        input_file (str): Path to the input CSV file
        output_file (str, optional): Path to the output CSV file. If None, overwrites the input file.
    """
    # Read the input CSV file
    df = pd.read_csv(input_file)
    
    # Create a new DataFrame with the same structure as synthetic_updated.csv
    new_df = pd.DataFrame()
    
    # Copy common columns
    common_columns = ['patient', 'note_id', 'note', 'document_timestamp', 'formatted_dates']
    for col in common_columns:
        if col in df.columns:
            new_df[col] = df[col]
    
    # Transform extracted_disorders to SNOMED and UMLS entities
    snomed_entities_list = []
    umls_entities_list = []
    
    for _, row in df.iterrows():
        snomed, umls = transform_extracted_disorders_to_snomed_umls(row['extracted_disorders'])
        snomed_entities_list.append(snomed)
        umls_entities_list.append(umls)
    
    new_df['extracted_snomed_entities'] = snomed_entities_list
    new_df['extracted_umls_entities'] = umls_entities_list
    
    # Add entity_gold column (empty for now)
    new_df['entity_gold'] = ['[]'] * len(df)
    
    # Transform relationship_gold
    new_df['relationship_gold'] = df['relationship_gold'].apply(transform_relationship_gold)
    
    # Add prediction columns
    empty_predictions = create_empty_prediction_columns()
    for col, default_value in empty_predictions.items():
        new_df[col] = [default_value] * len(df)
    
    # Save the transformed data
    if output_file:
        new_df.to_csv(output_file, index=False)
    else:
        new_df.to_csv(input_file, index=False)
    
    return new_df

if __name__ == "__main__":
    print("Transforming synthetic_multi.csv to match the structure of synthetic_updated.csv...")
    transform_multi_synthetic_data('data/synthetic_multi.csv')
    print("Transformation complete!")