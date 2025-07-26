import re
import os
import ast
from datetime import datetime
import json
import torch
from tqdm import tqdm
import pandas as pd
from utils.relative_date_utils import extract_relative_dates_llm

# Get the appropriate data path based on the config
def get_data_path(config):
    """
    Get the correct dataset path based on the config.DATA_SOURCE setting.
    
    Args:
        config: Configuration object with DATA_SOURCE and path attributes.
        
    Returns:
        str: Path to the dataset file.
    """
    data_source = config.DATA_SOURCE.lower()
    
    if data_source == 'synthetic':
        return config.SYNTHETIC_DATA_PATH
    elif data_source == 'synthetic_updated':
        return config.SYNTHETIC_UPDATED_DATA_PATH
    elif data_source == 'sample':
        return config.SAMPLE_DATA_PATH
    elif data_source == 'imaging':
        return config.IMAGING_DATA_PATH
    elif data_source == 'notes':
        return config.NOTES_DATA_PATH
    elif data_source == 'letters':
        return config.LETTERS_DATA_PATH
    elif data_source == 'nph':
        return config.NPH_DATA_PATH
    else:
        raise ValueError(f"Unknown data source: {data_source}. Valid options are: 'synthetic', 'synthetic_updated', 'sample', 'imaging', 'notes', 'letters', 'nph'")

def safe_json_loads(data_string):
    """
    Safely parses a JSON-like string.
    Tries json.loads first, then ast.literal_eval for single quotes and Python literals.
    Handles JSON's true/false/null for ast.literal_eval.

    Args:
        data_string (str): The string to parse.

    Returns:
        A Python object (e.g., dict, list).

    Raises:
        json.JSONDecodeError: if parsing fails with all methods.
    """
    if not isinstance(data_string, str):
        # To mimic json.loads, which raises TypeError for non-str/bytes/bytearray
        raise TypeError(f"the JSON object must be str, bytes or bytearray, not {type(data_string).__name__}")
    
    try:
        return json.loads(data_string)
    except json.JSONDecodeError as e:
        try:
            # ast.literal_eval can't handle 'true', 'false', 'null'. Let's replace them.
            s = re.sub(r'\btrue\b', 'True', data_string, flags=re.IGNORECASE)
            s = re.sub(r'\bfalse\b', 'False', s, flags=re.IGNORECASE)
            s = re.sub(r'\bnull\b', 'None', s)
            return ast.literal_eval(s)
        except (ValueError, SyntaxError, MemoryError) as e2:
            # If all else fails, raise the original error to be caught by the caller.
            raise json.JSONDecodeError(f"Failed to parse with all methods: {e2}", data_string, 0) from e

def transform_python_to_json(python_string):
    """
    Transform a Python-style string with single quotes and datetime objects 
    into a valid JSON string with double quotes.
    
    Args:
        python_string: A string representing Python objects (like a list of dicts with single quotes)
        
    Returns:
        A valid JSON string with all strings double-quoted
    """
    if not python_string or pd.isna(python_string):
        return "[]"  # Return empty JSON array if input is empty or NaN
        
    # Handle datetime.date objects by converting them to ISO format strings
    # Example: datetime.date(2019, 4, 18) -> "2019-04-18"
    date_pattern = r"datetime\.date\((\d+),\s*(\d+),\s*(\d+)\)"
    def date_replacer(match):
        year, month, day = map(int, match.groups())
        return f'"{year:04d}-{month:02d}-{day:02d}"'
    
    # Apply the datetime replacement
    python_string_cleaned = re.sub(date_pattern, date_replacer, python_string)
    
    try:
        # Use ast.literal_eval to safely parse the Python literal
        python_obj = ast.literal_eval(python_string_cleaned)
        
        # Convert to valid JSON string with double quotes
        return json.dumps(python_obj)
    except (SyntaxError, ValueError) as e:
        # If ast.literal_eval fails, return an empty JSON array
        print(f"Warning: Failed to parse Python-style string: {e}")
        return "[]"

# Clean and preprocess text for model input
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Replace special characters
    text = re.sub(r'[^\w\s\.]', ' ', text)
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def _parse_entity_column(data_string):
    """Helper to parse a JSON-like string from an entity column."""
    if not data_string or pd.isna(data_string):
        return []
    try:
        # First, ensure it's valid JSON, as it may be a Python literal string
        valid_json_string = transform_python_to_json(data_string)
        # Then, safely load the JSON
        entities = safe_json_loads(valid_json_string)
        
        # Process each entity to add a 'category' field based on 'categories'
        for entity in entities:
            # If entity has 'categories' but not 'category', add 'category' field
            if 'categories' in entity and not 'category' in entity:
                # Use the first category as the main category
                if isinstance(entity['categories'], list) and len(entity['categories']) > 0:
                    entity['category'] = entity['categories'][0].lower()
                else:
                    entity['category'] = 'unknown'
        
        return entities
    except (json.JSONDecodeError, TypeError) as e:
        print(f"Warning: Could not parse entity data string. Error: {e}. Data: '{str(data_string)[:100]}...'")
        return []

def load_and_prepare_data(dataset_path, num_samples, config=None):
    """
    Loads dataset, selects samples, prepares gold standard, and pre-extracts entities.
    Supports real data from CSV files.
    
    Args:
        dataset_path (str): Path to the dataset file (CSV). If None, will use config.DATA_SOURCE to determine path.
        num_samples (int): Maximum number of samples to use (if provided).
        config: Configuration object containing paths and column names.
        
    Returns:
        tuple: In multi_entity mode: (prepared_test_data, entity_gold, relationship_gold, pa_likelihood_gold) or (None, None, None, None) if loading fails.
               In disorder_only mode: (prepared_test_data, relationship_gold) with entity_gold and pa_likelihood_gold set to None.
               
               prepared_test_data is a list of dicts {'patient_id': ..., 'note_id': ..., 'note': ..., 'entities': ...}.
               entity_gold is a list of dicts {'note_id': ..., 'entity_label': ..., 'entity_category': ..., 'start': ..., 'end': ...}.
               relationship_gold is a list of dicts {'note_id': ..., 'patient_id': ..., 'entity_label': ..., 'entity_category': ..., 'date': ...}.
               pa_likelihood_gold is a dict mapping patient_id to likelihood value.
    """
    # Determine if we're in disorder_only mode
    entity_mode = getattr(config, 'ENTITY_MODE', 'multi_entity')
    disorder_only_mode = (entity_mode == 'disorder_only')
    
    # Print diagnostic information
    print(f"\n===== LOAD_AND_PREPARE_DATA =====")
    print(f"Entity mode: {entity_mode}")
    
        # If dataset_path is not provided and config is available, use config to determine path
    if dataset_path is None and config:
        if hasattr(config, 'DATA_SOURCE'):
            dataset_path = get_data_path(config)
        elif hasattr(config, 'DATA_PATH'):
            dataset_path = config.DATA_PATH
            
    if not dataset_path:
        print(f"Error: No dataset path provided and could not determine path from config")
        return (None, None) if disorder_only_mode else (None, None, None, None)
    
    text_column = config.REAL_DATA_TEXT_COLUMN
    patient_id_column = getattr(config, 'REAL_DATA_PATIENT_ID_COLUMN', None)
    
    # Get column names for gold standards
    entity_gold_col = getattr(config, 'ENTITY_GOLD_COLUMN', None)
    relationship_gold_col = getattr(config, 'RELATIONSHIP_GOLD_COLUMN', None)
    pa_likelihood_gold_col = getattr(config, 'PA_LIKELIHOOD_GOLD_COLUMN', None)
    legacy_gold_col = getattr(config, 'REAL_DATA_GOLD_COLUMN', None)
    
    # Determine which columns to use for entity extraction based on mode
    if disorder_only_mode:
        # In disorder_only mode, we use diagnoses_column for disorders
        diagnoses_column = getattr(config, 'REAL_DATA_DIAGNOSES_COLUMN', None)
        dates_column = getattr(config, 'REAL_DATA_DATES_COLUMN', None)
        print(f"Using disorder_only mode with columns: diagnoses={diagnoses_column}, dates={dates_column}")
    else:
        # In multi_entity mode, we use snomed_column and umls_column for entities
        snomed_column = getattr(config, 'REAL_DATA_SNOMED_COLUMN', None)
        umls_column = getattr(config, 'REAL_DATA_UMLS_COLUMN', None)
        dates_column = getattr(config, 'REAL_DATA_DATES_COLUMN', None)
        print(f"Using multi_entity mode with columns: snomed={snomed_column}, umls={umls_column}, dates={dates_column}")
    
    timestamp_column = getattr(config, 'REAL_DATA_TIMESTAMP_COLUMN', None)
    
    # Print available columns for debugging
    print(f"Dataset path: {dataset_path}")
    print(f"Text column: {text_column}")
    
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found at {dataset_path}")
        return (None, None) if disorder_only_mode else (None, None, None, None)
    
    print(f"Loading dataset from {dataset_path}...")
    try:
        # Read the CSV file
        df = pd.read_csv(dataset_path)
        
        # Print available columns for debugging
        print(f"Available columns in CSV: {list(df.columns)}")
        
        # Check if the text column exists
        if text_column not in df.columns:
            print(f"Error: Text column '{text_column}' not found in CSV. Available columns: {list(df.columns)}")
            return (None, None) if disorder_only_mode else (None, None, None, None)
            
        print(f"Found {len(df)} records in CSV.")
        
        # If a specific number of samples is requested, limit to that
        if num_samples and num_samples < len(df):
            df = df.iloc[:num_samples]
            print(f"Limiting to {num_samples} samples.")
        
        # Check if timestamp column exists (for relative date extraction)
        relative_date_extraction_enabled = False
        if hasattr(config, 'ENABLE_RELATIVE_DATE_EXTRACTION') and config.ENABLE_RELATIVE_DATE_EXTRACTION:
            if timestamp_column and timestamp_column in df.columns:
                relative_date_extraction_enabled = True
                print(f"Relative date extraction enabled using timestamp column: {timestamp_column}")
                
                # Add new column for storing LLM extracted dates
                df['llm_extracted_dates'] = None
            else:
                print(f"Warning: Relative date extraction enabled in config but timestamp column '{timestamp_column}' not found in CSV")
            
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return (None, None) if disorder_only_mode else (None, None, None, None)
    
    # Initialize gold standard data structures
    entity_gold = []
    relationship_gold = []
    pa_likelihood_gold = {}
    
    # --- Process Relationship Gold Standard ---
    # First check for the preferred column name
    gold_column = relationship_gold_col if relationship_gold_col and relationship_gold_col in df.columns else legacy_gold_col
    
    if gold_column and gold_column in df.columns:
        print(f"Found relationship gold standard column: {gold_column}. Processing gold standard data...")
        
        with tqdm(total=len(df), desc="Preparing relationship gold standard", unit="note") as pbar:
            for i, row in df.iterrows():
                # Check if the gold standard cell is not empty
                gold_data = row.get(gold_column)
                if pd.notna(gold_data) and gold_data:
                    try:
                        # Parse the JSON string in the gold standard column
                        gold_json = safe_json_loads(gold_data)
                        
                        # Check if this is the enhanced format (array of objects) or original format (object with 'relationships')
                        if isinstance(gold_json, list):
                            # Enhanced format - could have different structures
                            for entry in gold_json:
                                # Handle both formats: direct entity-date pairs or date-diagnoses structure
                                if 'entity_label' in entry and 'date' in entry:
                                    # Direct entity-date pair
                                    relationship_gold.append({
                                        'note_id': i,
                                        'patient_id': row.get(patient_id_column) if patient_id_column else None,
                                        'entity_label': str(entry['entity_label']).lower(),
                                        'entity_category': str(entry.get('entity_category', 'disorder')).lower(),
                                        'date': entry['date']
                                    })
                                elif 'diagnosis' in entry and 'date' in entry:
                                    # Legacy direct diagnosis-date pair
                                    relationship_gold.append({
                                        'note_id': i,
                                        'patient_id': row.get(patient_id_column) if patient_id_column else None,
                                        'entity_label': str(entry['diagnosis']).lower(),
                                        'entity_category': 'disorder',  # Default for legacy format
                                        'date': entry['date']
                                    })
                                elif 'date' in entry and 'diagnoses' in entry:
                                    # Date with multiple diagnoses
                                    date_str = entry.get('date')
                                    if not date_str:
                                        continue
                                    
                                    try:
                                        # Parse date string which might have non-padded month/day
                                        date_parts = str(date_str).split('-')
                                        if len(date_parts) == 3:
                                            year, month, day = date_parts
                                            normalized_date = f"{year}-{int(month):02d}-{int(day):02d}"
                                        else:
                                            normalized_date = date_str  # Keep original if not in expected format
                                    except (ValueError, TypeError):
                                        print(f"Warning: Could not normalize date '{date_str}' in gold standard for row {i}. Using as-is.")
                                        normalized_date = date_str
                                
                                # Process each diagnosis associated with this date
                                diagnoses = entry.get('diagnoses', [])
                                for diag in diagnoses:
                                    if 'diagnosis' in diag:
                                        relationship_gold.append({
                                            'note_id': i,
                                            'patient_id': row.get(patient_id_column) if patient_id_column else None,
                                            'entity_label': str(diag['diagnosis']).lower(),
                                                'entity_category': 'disorder',  # Default for legacy format
                                                'date': normalized_date
                                        })
                        # Original format with 'relationships' key
                        elif isinstance(gold_json, dict) and 'relationships' in gold_json and isinstance(gold_json['relationships'], list):
                            for rel in gold_json['relationships']:
                                if 'diagnosis' in rel and 'date' in rel:
                                    # Normalize the date from the gold standard
                                    date_str = rel.get('date')
                                    try:
                                        date_parts = str(date_str).split('-')
                                        if len(date_parts) == 3:
                                            year, month, day = date_parts
                                            normalized_date = f"{year}-{int(month):02d}-{int(day):02d}"
                                        else:
                                            normalized_date = date_str  # Keep original if not in expected format
                                    except (ValueError, TypeError):
                                        print(f"Warning: Could not normalize date '{date_str}' in legacy gold standard for row {i}. Using as-is.")
                                        normalized_date = date_str

                                    relationship_gold.append({
                                        'note_id': i,
                                        'patient_id': row.get(patient_id_column) if patient_id_column else None,
                                        'entity_label': str(rel['diagnosis']).lower(),
                                        'entity_category': 'disorder',  # Default for legacy format
                                        'date': normalized_date # Normalized YYYY-MM-DD format
                                    })
                        else:
                            print(f"Warning: Unrecognized gold standard format for row {i}")
                    except (json.JSONDecodeError, TypeError, KeyError) as e:
                        print(f"Warning: Could not parse relationship gold standard for row {i}: {e}")
                
                pbar.update(1)
        
        print(f"Prepared relationship gold standard with {len(relationship_gold)} relationships.")
    else:
        print("No relationship gold standard column found or specified. Evaluation metrics will not be calculated.")
    
    # --- Process Entity Gold Standard (only in multi_entity mode) ---
    if not disorder_only_mode and entity_gold_col and entity_gold_col in df.columns:
        print("Found entity gold standard column. Processing entity gold data...")
        
        with tqdm(total=len(df), desc="Preparing entity gold standard", unit="note") as pbar:
            for i, row in df.iterrows():
                # Check if the gold standard cell is not empty
                gold_data = row.get(entity_gold_col)
                if pd.notna(gold_data) and gold_data:
                    try:
                        # Parse the JSON string in the gold standard column
                        gold_json = safe_json_loads(gold_data)
                        
                        # Process each entity in the gold standard
                        for entity in gold_json:
                            entity_gold.append({
                                'note_id': i,
                                'patient_id': row.get(patient_id_column) if patient_id_column else None,
                                'entity_label': str(entity.get('entity_label', '')).lower(),
                                'entity_category': str(entity.get('entity_category', 'unknown')).lower(),
                                'start': entity.get('start', 0),
                                'end': entity.get('end', 0)
                            })
                    except (json.JSONDecodeError, TypeError, KeyError) as e:
                        print(f"Warning: Could not parse entity gold standard for row {i}: {e}")
                
                pbar.update(1)
        
        print(f"Prepared entity gold standard with {len(entity_gold)} entities.")
    elif not disorder_only_mode:
        print("No entity gold standard column found or specified.")
    
    # Pre-extract entities from the text column
    print("Pre-extracting entities...")
    prepared_test_data = []
    
    # Diagnostic counters
    notes_with_entities = 0
    total_entities = 0
    total_dates = 0
    
    with tqdm(total=len(df), desc="Processing annotations", unit="note") as pbar:
        for i, row in df.iterrows():
            # Get the text from the specified column
            text = str(row.get(text_column, ''))
            
            # This will hold the full dictionaries of extracted entities for NER evaluation
            all_extracted_entities = []
            
            # Initialize entities as empty lists in case we can't get valid annotations
            entities = ([], [])
            
            # Determine which entity extraction approach to use based on mode
            if disorder_only_mode:
                # --- DISORDER ONLY MODE ENTITY EXTRACTION ---
                # Check if we have pre-annotated entities in the CSV
                use_annotations = diagnoses_column and dates_column and diagnoses_column in df.columns and dates_column in df.columns
                
                # Comment out Row diagnostic information
                """
                if i < 3:  # Just for the first few rows
                    print(f"\nRow {i} - Using disorder_only annotations: {use_annotations}")
                    if use_annotations:
                        print(f"  Diagnoses column: {diagnoses_column}")
                        print(f"  Dates column: {dates_column}")
                """
                
                if use_annotations:
                    # Get the annotations from the CSV
                    diagnoses_data = row.get(diagnoses_column)
                    dates_data = row.get(dates_column)
                    
                    # Comment out raw data prints
                    """
                    # Print raw data for first few rows
                    if i < 3:
                        print(f"Row {i} - Raw diagnoses data: {str(diagnoses_data)[:100]}...")
                        print(f"Row {i} - Raw dates data: {str(dates_data)[:100]}...")
                    """
                    
                    if pd.notna(diagnoses_data) and pd.notna(dates_data):
                        try:
                            # Parse diagnoses annotations from JSON format
                            diagnoses_list = []
                            # Transform Python-style string to valid JSON before parsing
                            valid_diagnoses_json = transform_python_to_json(diagnoses_data)
                            disorders = safe_json_loads(valid_diagnoses_json)
                            for disorder in disorders:
                                label = disorder.get('label', '')
                                start_pos = disorder.get('start', 0)
                                diagnoses_list.append((label.lower(), start_pos))
                            
                            # Parse dates annotations from JSON format
                            dates_list = []
                            # Transform Python-style string to valid JSON before parsing
                            valid_dates_json = transform_python_to_json(dates_data)
                            formatted_dates = safe_json_loads(valid_dates_json)
                            for date_obj in formatted_dates:
                                parsed_date = date_obj.get('parsed', '')
                                original_date = date_obj.get('original', '')
                                start_pos = date_obj.get('start', 0)
                                dates_list.append((parsed_date, original_date, start_pos))
                            
                            # If we found entities, use them
                            if diagnoses_list or dates_list:
                                entities = (diagnoses_list, dates_list)
                                
                                # Update counters
                                if diagnoses_list and dates_list:
                                    notes_with_entities += 1
                                total_entities += len(diagnoses_list)
                                total_dates += len(dates_list)
                                
                                # Comment out entity debugging prints
                                """
                                if i < 3:  # Just for debugging, show first few entities
                                    print(f"Row {i} diagnoses: {diagnoses_list[:2]}...")
                                    print(f"Row {i} dates: {dates_list[:2]}...")
                                """
                        except Exception as e:
                            # Print detailed error
                            if i < 10:
                                print(f"ERROR parsing entities for row {i}: {str(e)}")
                                print(f"  diagnoses_data type: {type(diagnoses_data)}")
                                print(f"  dates_data type: {type(dates_data)}")
                                
                                # Try to show the problematic part of the data
                                if isinstance(diagnoses_data, str):
                                    print(f"  diagnoses_data sample: {diagnoses_data[:100]}...")
                                if isinstance(dates_data, str):
                                    print(f"  dates_data sample: {dates_data[:100]}...")
            else:
                # --- MULTI ENTITY MODE ENTITY EXTRACTION ---
                # In multi_entity mode, we load pre-annotated entities from CSV columns
                entities_list_for_rel_extraction = []
                dates_list = []
                
                # Initialize column variables
                snomed_column = config.REAL_DATA_SNOMED_COLUMN if hasattr(config, 'REAL_DATA_SNOMED_COLUMN') else None
                umls_column = config.REAL_DATA_UMLS_COLUMN if hasattr(config, 'REAL_DATA_UMLS_COLUMN') else None
                
                # Process SNOMED entities if available
                if snomed_column and snomed_column in df.columns:
                    snomed_data = row.get(snomed_column)
                    if pd.notna(snomed_data) and snomed_data:
                        try:
                            # Parse entities from the SNOMED column
                            snomed_entities = _parse_entity_column(snomed_data)
                            all_extracted_entities.extend(snomed_entities)
                        except Exception as e:
                            print(f"Error parsing SNOMED entities for row {i}: {e}")
                
                # Process UMLS entities if available
                if umls_column and umls_column in df.columns:
                    umls_data = row.get(umls_column)
                    if pd.notna(umls_data) and umls_data:
                        try:
                            # Parse entities from the UMLS column
                            umls_entities = _parse_entity_column(umls_data)
                            all_extracted_entities.extend(umls_entities)
                        except Exception as e:
                            print(f"Error parsing UMLS entities for row {i}: {e}")
                
                # Convert the full entity dicts to the (label, start_pos) tuple format
                # required by the downstream relationship extraction models.
                for entity in all_extracted_entities:
                    label = entity.get('label', '')
                    start_pos = entity.get('start', 0)
                    if label:
                        entities_list_for_rel_extraction.append((label.lower(), start_pos))
                
                # Process dates
                if dates_column and dates_column in df.columns:
                    dates_data = row.get(dates_column)
                    if pd.notna(dates_data) and dates_data:
                        try:
                            # Transform Python-style string to valid JSON before parsing
                            valid_dates_json = transform_python_to_json(dates_data)
                            formatted_dates = safe_json_loads(valid_dates_json)
                            for date_obj in formatted_dates:
                                parsed_date = date_obj.get('parsed', '')
                                original_date = date_obj.get('original', '')
                                start_pos = date_obj.get('start', 0)
                                dates_list.append((parsed_date, original_date, start_pos))
                        except Exception as e:
                            print(f"Error parsing dates for row {i}: {e}")
                
                # Update entities tuple with the combined data
                entities = (entities_list_for_rel_extraction, dates_list)
                
                # Update counters
                if entities_list_for_rel_extraction or dates_list:
                    notes_with_entities += 1
                total_entities += len(entities_list_for_rel_extraction)
                total_dates += len(dates_list)
            
            # Process relative dates if enabled (same for both modes)
            if relative_date_extraction_enabled:
                # Get the document timestamp
                timestamp_str = row.get(timestamp_column)
                
                if pd.notna(timestamp_str) and timestamp_str:
                    try:
                        # Parse the timestamp string into a datetime object
                        # Try different formats
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
                                document_timestamp = datetime.strptime(timestamp_str, format_str)
                                break
                            except ValueError:
                                continue
                        
                        if document_timestamp:
                            # Only print for a few rows to reduce output
                            if i % 20 == 0 or i < 3:
                                print(f"Extracting relative dates for row {i} using timestamp: {document_timestamp.strftime('%Y-%m-%d')}")
                            
                            # Extract relative dates using LLM
                            relative_dates = extract_relative_dates_llm(text, document_timestamp, config)
                            
                            if relative_dates:
                                # Append relative dates to existing dates list
                                entities_list_for_rel_extraction, dates_list = entities
                                combined_dates_list = dates_list + relative_dates
                                
                                # Update entities with combined dates
                                entities = (entities_list_for_rel_extraction, combined_dates_list)
                                
                                # Convert relative dates to JSON for storage in CSV
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
                                
                                if i % 20 == 0 or i < 3:  # Reduce output, just show some samples
                                    print(f"Row {i} extracted {len(relative_dates)} relative dates")
                    except Exception as e:
                        print(f"Error extracting relative dates for row {i}: {e}")
            
            # Add the prepared data entry - MOVED OUTSIDE THE IF/ELSE BLOCK
            prepared_test_data.append({
                'note_id': i,
                'patient_id': row.get(patient_id_column) if patient_id_column else None,
                'note': text,
                'entities': entities,
                'extracted_entities': all_extracted_entities  # Add this for NER evaluation
            })
            
            pbar.update(1)
    
    # Print summary of entity extraction
    print("\n===== ENTITY EXTRACTION SUMMARY =====")
    print(f"Total notes processed: {len(prepared_test_data)}")
    print(f"Notes with entities: {notes_with_entities}/{len(prepared_test_data)} ({notes_with_entities/len(prepared_test_data)*100:.1f}%)")
    print(f"Total entities extracted: {total_entities}")
    print(f"Total dates extracted: {total_dates}")
    print(f"Average entities per note: {total_entities/max(1, len(prepared_test_data)):.2f}")
    print(f"Average dates per note: {total_dates/max(1, len(prepared_test_data)):.2f}")
    print("===== END ENTITY EXTRACTION SUMMARY =====\n")
    
    # Save the updated dataframe back to CSV if we extracted relative dates
    if relative_date_extraction_enabled and 'llm_extracted_dates' in df.columns:
        try:
            # Create a backup of the original CSV
            backup_path = dataset_path + '.backup'
            if not os.path.exists(backup_path):
                df.to_csv(backup_path, index=False)
                print(f"Created backup of original CSV at {backup_path}")
            
            # Save the updated dataframe with LLM-extracted dates
            df.to_csv(dataset_path, index=False)
            print(f"Saved updated CSV with LLM-extracted dates to {dataset_path}")
        except Exception as e:
            print(f"Error saving updated CSV: {e}")
    
    # Return appropriate values based on mode
    if disorder_only_mode:
        return prepared_test_data, relationship_gold
    else:
        return prepared_test_data, entity_gold, relationship_gold, pa_likelihood_gold

def preprocess_note_for_prediction(note, diagnoses, dates, MAX_DISTANCE=500):
    """Preprocess features from a clinical note for prediction using pre-extracted entities.
    
    Args:
        note (str): The clinical note text
        diagnoses (list): List of (diagnosis, position) tuples
        dates (list): List of (parsed_date, original_date, position) tuples
        MAX_DISTANCE (int): Maximum distance between diagnosis and date to consider
        
    Returns:
        list: List of feature dictionaries for each diagnosis-date pair
    """
    # Import config here to avoid circular imports
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import config
    
    # Check if debug mode is enabled
    debug_mode = getattr(config, 'MODEL_DEBUG_MODE', False)
    
    # Build features for each diagnosis-date pair
    features = []
    pairs_considered = 0
    pairs_within_distance = 0
    
    for diagnosis, diag_pos in diagnoses:
        # Correctly unpack the 3-element tuple from the dates list
        for parsed_date, date_str, date_pos in dates:
            pairs_considered += 1
            distance = abs(diag_pos - date_pos)
            if distance > MAX_DISTANCE:
                continue
                
            pairs_within_distance += 1
            start_pos = max(0, min(diag_pos, date_pos) - 50)
            end_pos = min(len(note), max(diag_pos, date_pos) + 100)
            context = note[start_pos:end_pos]
            context = preprocess_text(context)
            
            # DIAGNOSTIC: Print detailed information about this candidate pair
            # Comment out the detailed candidate pair diagnostics to reduce verbosity
            """
            if debug_mode and pairs_within_distance <= 3:  # Only print first 3 for brevity
                print(f"\n--- Candidate Pair {pairs_within_distance} ---")
                print(f"Diagnosis: '{diagnosis}' at position {diag_pos}")
                print(f"Date: Original='{date_str}', Parsed='{parsed_date}' at position {date_pos}")
                print(f"Distance: {distance} words")
                print(f"Diagnosis before date: {'Yes' if diag_pos < date_pos else 'No'}")
                print(f"Context window: [{start_pos}:{end_pos}] (length: {end_pos-start_pos})")
                print(f"Raw context: '{note[start_pos:end_pos][:100]}...'")
                print(f"Preprocessed context: '{context[:100]}...'")
            """
            
            feature = {
                'diagnosis': diagnosis,
                'date': parsed_date, # Use parsed date instead of raw date string
                'context': context,
                'distance': distance,
                'diag_pos_rel': diag_pos - start_pos,
                'date_pos_rel': date_pos - start_pos,
                'diag_before_date': 1 if diag_pos < date_pos else 0
            }
            features.append(feature)
    
    # DIAGNOSTIC: Print combined input and summary statistics
    if debug_mode:
        print("\n===== DIAGNOSTIC: PREPROCESS_NOTE_FOR_PREDICTION =====")
        print(f"Number of diagnoses: {len(diagnoses)}")
        print(f"Number of dates: {len(dates)}")
        print(f"Max distance setting: {MAX_DISTANCE} words")
        print(f"Total diagnosis-date pairs considered: {pairs_considered}")
        print(f"Pairs within max distance ({MAX_DISTANCE} words): {pairs_within_distance}")
        print(f"Total features generated: {len(features)}")
        print("==========================================================\n")
    
    return features

def create_prediction_dataset(features, vocab, device, max_distance, max_context_len):
    """Convert preprocessed features into model-ready tensors
    
    Args:
        features (list): List of feature dictionaries
        vocab (Vocabulary): Vocabulary object with word2idx mapping
        device (torch.device): Device to place tensors on
        max_distance (int): Maximum distance value for normalization
        max_context_len (int): Maximum context length for padding/truncation
        
    Returns:
        list: List of tensor dictionaries ready for model input
    """
    # Import config here to avoid circular imports
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import config
    
    # Check if debug mode is enabled
    debug_mode = getattr(config, 'MODEL_DEBUG_MODE', False)
    
    test_data = []
    unknown_words_count = 0
    total_words_count = 0
    
    # Process first 3 features in detail for diagnostics
    detailed_diagnostics_count = min(3, len(features))
    
    for i, feature in enumerate(features):
        # For the first few features, print detailed diagnostics
        show_details = debug_mode and i < detailed_diagnostics_count
        
        # Comment out the detailed feature vectorization diagnostics to reduce verbosity
        """
        if show_details:
            print(f"\n--- Feature {i+1} Vectorization ---")
            print(f"Diagnosis: '{feature['diagnosis']}'")
            print(f"Date: '{feature['date']}'")
            print(f"First 50 words of context: '{' '.join(feature['context'].split()[:50])}...'")
        """
        
        # Convert words to indices
        context_indices = []
        feature_unknown_words = 0
        feature_total_words = 0
        
        for word in feature['context'].split():
            feature_total_words += 1
            total_words_count += 1
            
            if word in vocab.word2idx:
                context_indices.append(vocab.word2idx[word])
            else:
                context_indices.append(vocab.word2idx['<unk>'])
                feature_unknown_words += 1
                unknown_words_count += 1
        
        """
        if show_details:
            print(f"Words in context: {feature_total_words}")
            print(f"Unknown words: {feature_unknown_words} ({feature_unknown_words/feature_total_words*100:.1f}%)")
            print(f"First 10 word indices: {context_indices[:10]}...")
        """
        
        # Pad or truncate using max_context_len
        original_length = len(context_indices)
        if len(context_indices) > max_context_len:  
            context_indices = context_indices[:max_context_len]
            """
            if show_details:
                print(f"Context truncated from {original_length} to {max_context_len} tokens")
            """
        else:
            padding = [0] * (max_context_len - len(context_indices))
            context_indices.extend(padding)
            """
            if show_details:
                print(f"Context padded from {original_length} to {max_context_len} tokens")
            """
        
        # Normalize distance using max_distance
        distance = min(feature['distance'] / max_distance, 1.0)
        
        """
        if show_details:
            print(f"Raw distance: {feature['distance']}, Normalized: {distance:.4f}")
            print(f"Diagnosis before date: {feature['diag_before_date']}")
        """
        
        # Create tensor dict
        tensor_dict = {
            'context': torch.tensor(context_indices, dtype=torch.long).unsqueeze(0).to(device),
            'distance': torch.tensor(distance, dtype=torch.float).unsqueeze(0).to(device),
            'diag_before': torch.tensor(feature['diag_before_date'], dtype=torch.float).unsqueeze(0).to(device),
            'feature': feature  # Keep original feature for reference
        }
        test_data.append(tensor_dict)
    
    # DIAGNOSTIC: Print combined vectorization parameters and summary statistics
    if debug_mode:
        print("\n===== DIAGNOSTIC: CREATE_PREDICTION_DATASET =====")
        print(f"Vocabulary size: {vocab.n_words}")
        print(f"Max context length: {max_context_len}")
        print(f"Max distance for normalization: {max_distance}")
        print(f"Device: {device}")
        print(f"Total features processed: {len(features)}")
        print(f"Total words processed: {total_words_count}")
        print(f"Unknown words encountered: {unknown_words_count} ({unknown_words_count/total_words_count*100:.1f}% of total)")
        print(f"Test data tensors created: {len(test_data)}")
        print("==========================================================\n")
    
    return test_data

def predict_relationships(model, test_data):
    """Use the model to predict relationships between diagnoses and dates
    
    Args:
        model (DiagnosisDateRelationModel): Trained model
        test_data (list): List of tensor dictionaries
        
    Returns:
        list: List of relationship dictionaries with predictions
    """
    relationships = []
    model.eval()
    
    with torch.no_grad():
        for data in test_data:
            # Get prediction
            output = model(data['context'], data['distance'], data['diag_before'])
            prob = output.item()
            
            # Get original feature
            feature = data['feature']
            
            # Add to relationships if probability exceeds threshold
            if prob > 0.5:  # Can adjust threshold as needed
                relationships.append({
                    'diagnosis': feature['diagnosis'],
                    'date': feature['date'],
                    'confidence': prob
                })
    
    return relationships

# Helper function to run extraction process for a given extractor and data
def run_extraction(extractor, prepared_test_data):
    """
    Runs the extraction process for a given extractor on prepared data.

    Args:
        extractor: An initialized and loaded extractor object (subclass of BaseExtractor).
        prepared_test_data (list): List of dicts {'patient_id': ..., 'note_id': ..., 'note': ..., 'entities': ...}.

    Returns:
        list: List of predicted relationships [{'note_id': ..., 'patient_id': ..., 'entity_label': ..., 'entity_category': ..., 'date': ..., 'confidence': ...}]
             or [{'note_id': ..., 'patient_id': ..., 'diagnosis': ..., 'date': ..., 'confidence': ...}] in disorder_only mode.
    """
    print(f"Generating predictions using {extractor.name}...")
    all_predictions = []
    skipped_rels = 0
    
    # Determine if we're in disorder_only mode based on the extractor name
    # This is a heuristic - we assume extractors with 'naive', 'custom', or 'relcat' in their name
    # are older extractors that use the 'diagnosis' field instead of 'entity_label'
    disorder_only_extractors = ['naive', 'custom', 'relcat']
    disorder_only_mode = any(name.lower() in extractor.name.lower() for name in disorder_only_extractors)
    
    for note_entry in tqdm(prepared_test_data, desc=f"Processing with {extractor.name}", unit="note"):
        note_id = note_entry['note_id']
        patient_id = note_entry['patient_id']
        try:
            # Extract relationships using the provided extractor
            relationships = extractor.extract(note_entry['note'], entities=note_entry['entities'], 
                                            note_id=note_id, patient_id=patient_id)
            
            for rel in relationships:
                # Check if we have a disorder_only mode extractor (using 'diagnosis')
                # or a multi_entity mode extractor (using 'entity_label')
                if 'diagnosis' in rel:
                    # disorder_only mode
                    entity_label = rel.get('diagnosis')
                    entity_category = 'disorder'  # Default for disorder_only mode
                elif 'entity_label' in rel:
                    # multi_entity mode
                    entity_label = rel.get('entity_label')
                    entity_category = rel.get('entity_category', 'unknown')
                else:
                    # Missing both required fields
                    print(f"Warning: Relationship missing both 'diagnosis' and 'entity_label' fields: {rel}")
                    skipped_rels += 1
                    continue
                
                # Get the date
                date_str = rel.get('date')
                
                if date_str is None or entity_label is None:
                    missing_fields = []
                    if date_str is None:
                        missing_fields.append("'date'")
                    if entity_label is None:
                        missing_fields.append("'diagnosis'/'entity_label'")
                    
                    print(f"Warning: Skipping relationship in note {note_id} due to missing {' and '.join(missing_fields)}. Rel: {rel}")
                    skipped_rels += 1
                    continue

                # Normalize entity label and date
                normalized_entity = str(entity_label).strip().lower() # Ensure string, strip, lower
                date_str = str(date_str).strip() # Ensure string and strip whitespace

                if date_str and normalized_entity:
                    # Create prediction in the appropriate format based on the mode
                    if disorder_only_mode:
                        all_predictions.append({
                            'note_id': note_id,
                            'patient_id': patient_id,
                            'diagnosis': normalized_entity,
                            'date': date_str,
                            'confidence': rel.get('confidence', 1.0) # Default confidence to 1.0 if missing
                        })
                    else:
                        all_predictions.append({
                            'note_id': note_id,
                            'patient_id': patient_id,
                            'entity_label': normalized_entity,
                            'entity_category': entity_category,
                            'date': date_str,
                            'confidence': rel.get('confidence', 1.0) # Default confidence to 1.0 if missing
                        })
                else:
                    # Log if validation failed
                    skipped_rels += 1
        except Exception as e:
            # Log errors during extraction for a specific note
            print(f"Extraction error on note {note_id} for {extractor.name}: {e}")
            # Optionally, re-raise if you want errors to halt execution: raise e
            continue # Continue with the next note

    print(f"Generated {len(all_predictions)} predictions. Skipped {skipped_rels} potentially invalid relationships.")
    return all_predictions