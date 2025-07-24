import pandas as pd
import json
import re
import os
from datetime import datetime

def extract_dates_from_text(text):
    """
    Extract all dates from the text with their positions.
    
    Args:
        text (str): The clinical note text.
        
    Returns:
        list: List of (date_str, start_pos, end_pos) tuples.
    """
    # Find all date patterns in the text
    date_patterns = [
        # Standard formats with parentheses
        r'\((\d{1,2})(?:st|nd|rd|th)?\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}\)',  # (30nd Jun 2024)
        r'\((\d{1,2})(?:st|nd|rd|th)?[-/\.](\d{1,2})[-/\.](\d{2,4})\)',  # (17.12.24) or (12/02/25)
        r'\(\d{4}-\d{1,2}-\d{1,2}\)',  # (2024-10-04)
        r'\((\d{1,2})\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[\']\d{2}\)',  # (16 Sep'24)
        
        # Standard formats without parentheses
        r'(?<!\()\b(\d{1,2})(?:st|nd|rd|th)?\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}\b',  # 30nd Jun 2024
        r'(?<!\()\b(\d{1,2})(?:st|nd|rd|th)?[-/\.](\d{1,2})[-/\.](\d{2,4})\b',  # 17.12.24 or 12/02/25
        r'(?<!\()\b\d{4}-\d{1,2}-\d{1,2}\b',  # 2024-10-04
        r'(?<!\()\b(\d{1,2})\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[\']\d{2}\b',  # 16 Sep'24
    ]
    
    date_matches = []
    
    # Search for each pattern
    for pattern in date_patterns:
        for match in re.finditer(pattern, text):
            date_str = match.group(0)
            start_pos = match.start()
            end_pos = match.end()
            date_matches.append((date_str, start_pos, end_pos))
    
    # Sort by position
    date_matches.sort(key=lambda x: x[1])
    return date_matches

def parse_date(date_str):
    """
    Parse a date string into a standardized YYYY-MM-DD format.
    
    Args:
        date_str (str): The date string to parse.
        
    Returns:
        str: The parsed date in YYYY-MM-DD format.
    """
    # Remove parentheses if present
    clean_date_str = date_str.strip('()')
    
    # Handle common patterns directly with regex
    
    # Pattern: 30nd Jun 2024 or 30 Jun 2024
    month_names = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6, 
                  'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}
    
    # Pattern: day month year (30nd Jun 2024)
    pattern1 = re.compile(r'(\d{1,2})(?:st|nd|rd|th)?\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+(\d{4})', re.IGNORECASE)
    match = pattern1.search(clean_date_str)
    if match:
        day = int(match.group(1))
        month = month_names[match.group(2).lower()[:3]]
        year = int(match.group(3))
        return f"{year:04d}-{month:02d}-{day:02d}"
    
    # Pattern: day month'year (16 Sep'24)
    pattern2 = re.compile(r'(\d{1,2})(?:st|nd|rd|th)?\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[\'"](\d{2})', re.IGNORECASE)
    match = pattern2.search(clean_date_str)
    if match:
        day = int(match.group(1))
        month = month_names[match.group(2).lower()[:3]]
        year = 2000 + int(match.group(3))  # Assume 20xx for 2-digit years
        return f"{year:04d}-{month:02d}-{day:02d}"
    
    # Pattern: dd.mm.yy or dd/mm/yy or dd-mm-yy
    pattern3 = re.compile(r'(\d{1,2})(?:st|nd|rd|th)?[-/\.](\d{1,2})[-/\.](\d{2,4})')
    match = pattern3.search(clean_date_str)
    if match:
        day = int(match.group(1))
        month = int(match.group(2))
        year = int(match.group(3))
        if year < 100:
            year += 2000  # Assume 20xx for 2-digit years
        return f"{year:04d}-{month:02d}-{day:02d}"
    
    # Pattern: yyyy-mm-dd
    pattern4 = re.compile(r'(\d{4})-(\d{1,2})-(\d{1,2})')
    match = pattern4.search(clean_date_str)
    if match:
        year = int(match.group(1))
        month = int(match.group(2))
        day = int(match.group(3))
        return f"{year:04d}-{month:02d}-{day:02d}"
    
    # If none of the patterns match, try using the relationship_gold data
    print(f"Warning: Could not parse date '{date_str}' with custom patterns")
    
    # Try standard datetime parsing as a fallback
    try:
        # Replace ordinal indicators to help parsing
        clean_date_str = re.sub(r'(\d+)(st|nd|rd|th)', r'\1', clean_date_str)
        
        # Try different formats
        formats = [
            '%d %b %Y',          # 30 Jun 2024
            '%d-%m-%Y',          # 30-06-2024
            '%d/%m/%Y',          # 30/06/2024
            '%d.%m.%Y',          # 30.06.2024
            '%Y-%m-%d',          # 2024-06-30
            '%d %b\'%y',         # 30 Jun'24
            '%d-%m-%y',          # 30-06-24
            '%d/%m/%y',          # 30/06/24
            '%d.%m.%y',          # 30.06.24
        ]
        
        for fmt in formats:
            try:
                dt = datetime.strptime(clean_date_str, fmt)
                return dt.strftime('%Y-%m-%d')
            except ValueError:
                continue
    except Exception as e:
        print(f"Error parsing date with datetime: {e}")
    
    # If all parsing attempts fail
    print(f"Failed to parse date: {date_str}")
    return None

def find_entity_in_text(text, entity_label):
    """
    Find all occurrences of an entity in the text.
    
    Args:
        text (str): The text to search in.
        entity_label (str): The entity label to find.
        
    Returns:
        list: List of (start_pos, end_pos) tuples.
    """
    positions = []
    start_pos = 0
    
    # Replace underscores with spaces for matching
    search_term = entity_label.replace('_', ' ')
    
    while True:
        pos = text.lower().find(search_term.lower(), start_pos)
        if pos == -1:
            break
        positions.append((pos, pos + len(search_term)))
        start_pos = pos + 1
    
    # If no matches found with spaces, try with underscores
    if not positions:
        start_pos = 0
        while True:
            pos = text.lower().find(entity_label.lower(), start_pos)
            if pos == -1:
                break
            positions.append((pos, pos + len(entity_label)))
            start_pos = pos + 1
    
    return positions

def remove_duplicate_entities(entities_list):
    """
    Remove duplicate entities from the list based on label and position.
    
    Args:
        entities_list (list): List of entity dictionaries.
        
    Returns:
        list: List of unique entity dictionaries.
    """
    # Create a set of (label, start, end) tuples to track unique entities
    unique_entities = set()
    result = []
    
    for entity in entities_list:
        # Create a tuple of the key fields
        entity_key = (entity.get('label', ''), entity.get('start', 0), entity.get('end', 0))
        
        # Only add if not already in the set
        if entity_key not in unique_entities:
            unique_entities.add(entity_key)
            result.append(entity)
    
    return result

def remove_duplicate_diagnoses(diagnoses_list):
    """
    Remove duplicate diagnoses from the list based on diagnosis and position.
    
    Args:
        diagnoses_list (list): List of diagnosis dictionaries.
        
    Returns:
        list: List of unique diagnosis dictionaries.
    """
    # Create a set of (diagnosis, position) tuples to track unique diagnoses
    unique_diagnoses = set()
    result = []
    
    for diag in diagnoses_list:
        # Create a tuple of the key fields
        diag_key = (diag.get('diagnosis', ''), diag.get('position', 0))
        
        # Only add if not already in the set
        if diag_key not in unique_diagnoses:
            unique_diagnoses.add(diag_key)
            result.append(diag)
    
    return result

def rebuild_synthetic_data(input_file, output_file=None):
    """
    Rebuild the synthetic data with corrected positions.
    
    Args:
        input_file (str): Path to the input CSV file.
        output_file (str, optional): Path to the output CSV file. If None, overwrites the input file.
    """
    print(f"Reading data from {input_file}...")
    df = pd.read_csv(input_file)
    
    print(f"Found {len(df)} records. Processing...")
    
    # Process each row
    for i, row in df.iterrows():
        if i % 10 == 0:
            print(f"Processing row {i}/{len(df)}...")
        
        # Get the note text
        text = row['note']
        
        # Extract dates from the text
        date_matches = extract_dates_from_text(text)
        
        # Process gold standard data
        gold_data = []
        if pd.notna(row['relationship_gold']):
            try:
                gold_json = json.loads(row['relationship_gold'])
                cleaned_gold_json = []
                
                # Verify and correct gold standard positions
                for item in gold_json:
                    date_str = item.get('date')
                    date_position = item.get('date_position')
                    diagnoses = item.get('diagnoses', [])
                    
                    # Skip items with no diagnoses
                    if not diagnoses:
                        print(f"Row {i}: Removing relationship with no diagnoses for date {date_str}")
                        continue
                    
                    # Find the actual date position in the text
                    actual_date_positions = []
                    for date_match, start_pos, end_pos in date_matches:
                        parsed_date = parse_date(date_match)
                        if parsed_date == date_str:
                            actual_date_positions.append((start_pos, end_pos))
                    
                    # Update date position if needed
                    if actual_date_positions:
                        # Find the closest position to the original
                        closest_pos = min(actual_date_positions, key=lambda x: abs(x[0] - date_position))
                        if date_position != closest_pos[0]:
                            print(f"Row {i}: Correcting date position for {date_str} from {date_position} to {closest_pos[0]}")
                            item['date_position'] = closest_pos[0]
                    
                    # Process diagnoses
                    for diag in diagnoses:
                        diagnosis = diag.get('diagnosis')
                        position = diag.get('position')
                        
                        # Find the actual entity position in the text
                        entity_positions = find_entity_in_text(text, diagnosis)
                        
                        # Update entity position if needed
                        if entity_positions:
                            # Find the closest position to the original
                            closest_pos = min(entity_positions, key=lambda x: abs(x[0] - position))
                            if position != closest_pos[0]:
                                print(f"Row {i}: Correcting position for {diagnosis} from {position} to {closest_pos[0]}")
                                diag['position'] = closest_pos[0]
                    
                    # Deduplicate diagnoses
                    unique_diagnoses = remove_duplicate_diagnoses(diagnoses)
                    if len(unique_diagnoses) < len(diagnoses):
                        print(f"Row {i}: Removed {len(diagnoses) - len(unique_diagnoses)} duplicate diagnoses for date {date_str}")
                        item['diagnoses'] = unique_diagnoses
                    
                    # Only add items with diagnoses
                    if unique_diagnoses:
                        cleaned_gold_json.append(item)
                
                gold_data = cleaned_gold_json
            except Exception as e:
                print(f"Error processing gold data for row {i}: {e}")
        
        # Update the gold standard data
        if gold_data:
            df.at[i, 'relationship_gold'] = json.dumps(gold_data)
        
        # Process extracted disorders
        disorders_data = []
        if pd.notna(row['extracted_disorders']):
            try:
                disorders_json = json.loads(row['extracted_disorders'])
                
                # Verify and correct disorder positions
                for item in disorders_json:
                    label = item.get('label')
                    start_pos = item.get('start')
                    end_pos = item.get('end')
                    
                    # Find the actual entity position in the text
                    entity_positions = find_entity_in_text(text, label)
                    
                    # Update entity position if needed
                    if entity_positions:
                        # Find the closest position to the original
                        closest_pos = min(entity_positions, key=lambda x: abs(x[0] - start_pos))
                        if start_pos != closest_pos[0] or end_pos != closest_pos[1]:
                            print(f"Row {i}: Correcting position for {label} from {start_pos}-{end_pos} to {closest_pos[0]}-{closest_pos[1]}")
                            item['start'] = closest_pos[0]
                            item['end'] = closest_pos[1]
                
                # Remove duplicate entities
                unique_disorders = remove_duplicate_entities(disorders_json)
                if len(unique_disorders) < len(disorders_json):
                    print(f"Row {i}: Removed {len(disorders_json) - len(unique_disorders)} duplicate entities")
                
                disorders_data = unique_disorders
            except Exception as e:
                print(f"Error processing disorders data for row {i}: {e}")
        
        # Update the extracted disorders data
        if disorders_data:
            df.at[i, 'extracted_disorders'] = json.dumps(disorders_data)
        
        # Process formatted dates
        formatted_dates = []
        for date_str, start_pos, end_pos in date_matches:
            # Parse the date
            parsed_date = parse_date(date_str)
            
            if parsed_date:
                formatted_dates.append({
                    'original': date_str,
                    'parsed': parsed_date,
                    'start': start_pos,
                    'end': end_pos
                })
        
        # Update the formatted_dates column
        df.at[i, 'formatted_dates'] = json.dumps(formatted_dates)
    
    # Save the updated dataframe
    if output_file:
        print(f"Writing updated data to {output_file}...")
        df.to_csv(output_file, index=False)
    else:
        print(f"Updating {input_file} in place...")
        temp_file = input_file + ".tmp"
        df.to_csv(temp_file, index=False)
        
        # Replace the original file with the temporary file
        import shutil
        shutil.move(temp_file, input_file)
    
    print("Done!")

if __name__ == "__main__":
    input_file = "data/synthetic.csv"
    
    # Create a backup of the original file
    backup_file = "data/synthetic_backup.csv"
    if not os.path.exists(backup_file):
        print(f"Creating backup of original file at {backup_file}...")
        import shutil
        shutil.copy2(input_file, backup_file)
    
    # Rebuild the synthetic data
    rebuild_synthetic_data(input_file)
    
    print("All done!") 