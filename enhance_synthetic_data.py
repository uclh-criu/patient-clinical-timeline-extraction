#!/usr/bin/env python3
# enhance_synthetic_data.py - Enhance synthetic data to match real-world data statistics
# 
# This script modifies synthetic data to make its statistical properties
# more closely match those of real-world clinical data.

import os
import sys
import json
import pandas as pd
import numpy as np
import random
import re
import ast
from copy import deepcopy
import argparse
from tqdm import tqdm

# ===== CONFIGURABLE VARIABLES =====

# Target statistics for diagnosis_only mode (imaging dataset)
DIAGNOSIS_ONLY_TARGETS = {
    'avg_document_length': 1260,  # characters
    'avg_entities_per_document': 5.8,
    'avg_dates_per_document': 2.7,
    'avg_entity_date_distance': 254,
    'class_imbalance_ratio': 3.4
}

# Target statistics for multi_entity mode (nph dataset)
MULTI_ENTITY_TARGETS = {
    'avg_document_length': 2230,  # characters
    'avg_entities_per_document': 25,  # Target is 25 SNOMED entities (will result in ~50 total entities with UMLS)
    'avg_dates_per_document': 5.0,
    'avg_entity_date_distance': 205,
    'class_imbalance_ratio': 17.2
}

# Clinical filler text snippets to insert into notes
FILLER_TEXT = [
    # Standard clinical phrases
    "The patient's vital signs are stable.",
    "A review of systems was otherwise negative.",
    "He denies any fever, chills, or night sweats.",
    "She denies any nausea, vomiting, or diarrhea.",
    "Past medical history is non-contributory.",
    "The plan was discussed with the patient, who understands and agrees.",
    "Labs from this morning were unremarkable.",
    "Follow-up appointment scheduled in 3 months.",
    "Patient tolerated the procedure well.",
    "No acute distress noted on examination.",
    "Lungs clear to auscultation bilaterally.",
    "Heart: Regular rate and rhythm, no murmurs.",
    "Abdomen: Soft, non-tender, non-distended.",
    "Neurological exam within normal limits.",
    "Patient is afebrile with normal vital signs.",
    "The patient was advised to return if symptoms worsen.",
    "Medications were reconciled during the visit.",
    "Patient reports compliance with current medication regimen.",
    "Imaging studies were reviewed with the patient.",
    "The patient was counseled on lifestyle modifications.",
    
    # More detailed clinical observations
    "HEENT: Normocephalic, atraumatic. Pupils equal, round, and reactive to light.",
    "Cardiovascular: Regular rate and rhythm. No murmurs, rubs, or gallops. Normal S1 and S2.",
    "Respiratory: Clear to auscultation bilaterally. No wheezes, rales, or rhonchi.",
    "Musculoskeletal: Full range of motion in all extremities. No edema or deformity noted.",
    "Skin: No rashes, lesions, or abnormal pigmentation. Good turgor and hydration.",
    "Psychiatric: Alert and oriented x3. Mood and affect appropriate.",
    "Lymphatic: No cervical, axillary, or inguinal lymphadenopathy.",
    "GI: Bowel sounds present in all four quadrants. No hepatosplenomegaly.",
    
    # Medication-related text
    "Current medications include: lisinopril 10mg daily, metformin 500mg twice daily, and atorvastatin 20mg at bedtime.",
    "Patient reports occasional use of over-the-counter analgesics for pain management.",
    "Medication dosages were adjusted based on recent laboratory findings.",
    "Patient was instructed on proper medication administration techniques.",
    "No known drug allergies or adverse reactions reported.",
    
    # Laboratory and imaging findings
    "Complete blood count shows WBC 7.2, Hgb 13.5, Plt 250.",
    "Basic metabolic panel within normal limits with sodium 140, potassium 4.2, creatinine 0.9.",
    "Liver function tests show mild elevation in ALT and AST, likely due to medication effect.",
    "Urinalysis negative for blood, protein, and leukocyte esterase.",
    "CT scan of the head shows no acute intracranial abnormalities.",
    "MRI of the brain demonstrates age-appropriate atrophy without evidence of mass or stroke.",
    "Chest X-ray reveals clear lung fields without infiltrates or effusions.",
    "EKG shows normal sinus rhythm without ST changes or conduction abnormalities.",
    
    # Plan and follow-up text
    "Will continue current medication regimen and reassess in 3 months.",
    "Patient instructed to follow up with specialist for further evaluation.",
    "Recommended lifestyle modifications including diet changes and regular exercise.",
    "Patient given educational materials regarding disease management.",
    "Referral placed for physical therapy twice weekly for 6 weeks.",
    "Will order additional laboratory studies at next visit if symptoms persist.",
    "Patient advised to maintain a symptom diary for review at next appointment."
]

# Top entities from the imaging dataset (diagnosis_only mode)
TOP_DIAGNOSIS_ENTITIES = [
    # Original top entities from imaging dataset
    "congenital malformation",
    "cerebral hemorrhage",
    "pituitary macroadenoma",
    "disorder of optic nerve",
    "hydrocephalus",
    "disorder of pituitary gland",
    "pituitary adenoma",
    "cyst",
    "pneumocephalus",
    "neoplasm of pituitary gland",
    "disorder of liver",
    "disorder of pancreas",
    "intracranial meningioma",
    "soft tissue lesion",
    "internal carotid artery stenosis",
    "meningitis",
    "radiologic infiltrate of lung",
    "cerebrovascular accident",
    "ebola virus disease",
    "synovitis",
    
    # Additional generic medical conditions
    "hypertension",
    "type 2 diabetes",
    "migraine",
    "asthma",
    "chronic kidney disease",
    "hypothyroidism",
    "hyperlipidemia",
    "osteoarthritis",
    "depression",
    "anxiety disorder",
    "gastroesophageal reflux disease",
    "atrial fibrillation",
    "chronic obstructive pulmonary disease",
    "obesity",
    "coronary artery disease",
    "heart failure",
    "osteoporosis",
    "anemia",
    "vitamin D deficiency",
    "sleep apnea"
]

# Top entities from the nph dataset (multi_entity mode)
TOP_SNOMED_ENTITIES = [
    # Original top entities from NPH dataset
    ("gait function observable entity", "finding"),
    ("normal pressure hydrocephalus", "disorder"),
    ("falls", "finding"),
    ("medical history", "finding"),
    ("hospital admission", "procedure"),
    ("problem", "finding"),
    ("drainage procedure", "procedure"),
    ("mobility as a finding", "finding"),
    ("ventriculoperitoneal shunt", "therapeutic or preventive procedure"),
    ("in care", "finding"),
    ("urinary incontinence", "pathologic function"),
    ("balance", "pharmacologic substance"),
    ("symptoms", "sign or symptom"),
    ("creation of shunt", "therapeutic or preventive procedure"),
    ("worse", "finding"),
    ("construction of shunt", "procedure"),
    ("cardiac shunt", "finding"),
    ("present", "finding"),
    ("chief complaint", "finding"),
    ("dementia", "mental or behavioral dysfunction"),
    
    # Additional generic findings
    ("headache", "finding"),
    ("dizziness", "finding"),
    ("fatigue", "finding"),
    ("nausea", "finding"),
    ("vomiting", "finding"),
    ("chest pain", "finding"),
    ("shortness of breath", "finding"),
    
    # Additional generic disorders
    ("hypertension", "disorder"),
    ("diabetes mellitus", "disorder"),
    ("asthma", "disorder"),
    ("chronic obstructive pulmonary disease", "disorder"),
    ("coronary artery disease", "disorder"),
    
    # Additional procedures
    ("magnetic resonance imaging", "procedure"),
    ("computed tomography", "procedure"),
    ("blood test", "procedure"),
    ("physical examination", "procedure"),
    
    # Additional medications
    ("acetaminophen", "pharmacologic substance"),
    ("ibuprofen", "pharmacologic substance"),
    ("lisinopril", "pharmacologic substance"),
    ("metformin", "pharmacologic substance"),
    ("atorvastatin", "pharmacologic substance")
]

# Using the same entities for UMLS with equivalent categories
TOP_UMLS_ENTITIES = [
    # Original top entities from NPH dataset
    ("gait function observable entity", "Finding"),
    ("normal pressure hydrocephalus", "Disease or Syndrome"),
    ("falls", "Finding"),
    ("medical history", "Finding"),
    ("hospital admission", "Health Care Activity"),
    ("problem", "Finding"),
    ("drainage procedure", "Therapeutic or Preventive Procedure"),
    ("mobility as a finding", "Finding"),
    ("ventriculoperitoneal shunt", "Therapeutic or Preventive Procedure"),
    ("in care", "Finding"),
    ("urinary incontinence", "Pathologic Function"),
    ("balance", "Pharmacologic Substance"),
    ("symptoms", "Sign or Symptom"),
    ("creation of shunt", "Therapeutic or Preventive Procedure"),
    ("worse", "Finding"),
    ("construction of shunt", "Therapeutic or Preventive Procedure"),
    ("cardiac shunt", "Finding"),
    ("present", "Finding"),
    ("chief complaint", "Finding"),
    ("dementia", "Mental or Behavioral Dysfunction"),
    
    # Additional generic findings
    ("headache", "Sign or Symptom"),
    ("dizziness", "Sign or Symptom"),
    ("fatigue", "Sign or Symptom"),
    ("nausea", "Sign or Symptom"),
    ("vomiting", "Sign or Symptom"),
    ("chest pain", "Sign or Symptom"),
    ("shortness of breath", "Sign or Symptom"),
    
    # Additional generic disorders
    ("hypertension", "Disease or Syndrome"),
    ("diabetes mellitus", "Disease or Syndrome"),
    ("asthma", "Disease or Syndrome"),
    ("chronic obstructive pulmonary disease", "Disease or Syndrome"),
    ("coronary artery disease", "Disease or Syndrome"),
    
    # Additional procedures
    ("magnetic resonance imaging", "Diagnostic Procedure"),
    ("computed tomography", "Diagnostic Procedure"),
    ("blood test", "Laboratory Procedure"),
    ("physical examination", "Diagnostic Procedure"),
    
    # Additional medications
    ("acetaminophen", "Pharmacologic Substance"),
    ("ibuprofen", "Pharmacologic Substance"),
    ("lisinopril", "Pharmacologic Substance"),
    ("metformin", "Pharmacologic Substance"),
    ("atorvastatin", "Pharmacologic Substance")
]

# ===== HELPER FUNCTIONS =====

def parse_json_field(field):
    """Parse a JSON field from the DataFrame."""
    if isinstance(field, str):
        try:
            return json.loads(field)
        except:
            try:
                return ast.literal_eval(field)
            except:
                return []
    return field

def insert_text_at_position(text, insert_text, position):
    """Insert text at the specified position and return the new text."""
    return text[:position] + insert_text + text[position:]

def recalculate_positions(entities, insert_position, insert_length):
    """
    Recalculate entity positions after text insertion.
    
    Args:
        entities: List of entity dictionaries with 'start' and 'end' positions
        insert_position: Position where text was inserted
        insert_length: Length of inserted text
    
    Returns:
        Updated entities list with recalculated positions
    """
    updated_entities = []
    
    for entity in entities:
        entity_copy = deepcopy(entity)
        
        # Update start position if it's after the insertion point
        if entity_copy['start'] >= insert_position:
            entity_copy['start'] += insert_length
        
        # Update end position if it's after the insertion point
        if entity_copy['end'] >= insert_position:
            entity_copy['end'] += insert_length
        
        updated_entities.append(entity_copy)
    
    return updated_entities

def update_relationship_positions(relationships, entity_map, date_map):
    """
    Update positions in relationship_gold based on new entity and date positions.
    
    Args:
        relationships: List of relationship dictionaries
        entity_map: Dictionary mapping entity labels to their new positions
        date_map: Dictionary mapping dates to their new positions
    
    Returns:
        Updated relationships list with recalculated positions
    """
    updated_relationships = []
    
    for rel in relationships:
        rel_copy = deepcopy(rel)
        
        # Handle diagnosis_only mode
        if 'diagnoses' in rel_copy:
            date = rel_copy.get('date', '')
            if date in date_map:
                rel_copy['date_position'] = date_map[date]
            
            updated_diagnoses = []
            for diag in rel_copy.get('diagnoses', []):
                diag_copy = deepcopy(diag)
                diagnosis = diag_copy.get('diagnosis', '')
                if diagnosis in entity_map:
                    diag_copy['position'] = entity_map[diagnosis]
                updated_diagnoses.append(diag_copy)
            
            rel_copy['diagnoses'] = updated_diagnoses
        
        # Handle multi_entity mode
        elif 'entity_label' in rel_copy:
            entity_label = rel_copy.get('entity_label', '')
            if entity_label in entity_map:
                rel_copy['entity_position'] = entity_map[entity_label]
        
        updated_relationships.append(rel_copy)
    
    return updated_relationships

def reduce_date_density(dates, relationships, target_avg_dates):
    """
    Randomly removes dates and their associated relationships to meet the
    target average.
    
    Args:
        dates: List of date dictionaries
        relationships: List of relationship dictionaries
        target_avg_dates: Target average number of dates per document
    
    Returns:
        Tuple of (surviving_dates, surviving_relationships)
    """
    num_dates_to_remove = len(dates) - int(target_avg_dates)
    
    if num_dates_to_remove <= 0:
        return dates, relationships
    
    # Select dates to remove
    dates_to_remove = random.sample(dates, k=num_dates_to_remove)
    removed_date_strings = {d['parsed'] for d in dates_to_remove}
    
    # Filter the dates list
    surviving_dates = [d for d in dates if d['parsed'] not in removed_date_strings]
    
    # Filter the relationships list
    surviving_relationships = []
    for rel in relationships:
        if rel.get('date') not in removed_date_strings:
            surviving_relationships.append(rel)
    
    return surviving_dates, surviving_relationships

def increase_positive_pair_distance(note_text, entities, dates, relationships, entity_mode):
    """
    Finds positive entity-date pairs and inserts filler text between them
    to increase their character distance.
    
    Args:
        note_text: The original note text
        entities: List of entity dictionaries
        dates: List of date dictionaries
        relationships: List of relationship dictionaries
        entity_mode: 'diagnosis_only' or 'multi_entity'
    
    Returns:
        Tuple of (updated_note, updated_entities, updated_dates)
    """
    # Create a quick lookup map for entity and date positions
    entity_positions = {e['label']: (e['start'], e['end']) for e in entities}
    date_positions = {d['parsed']: (d['start'], d['end']) for d in dates}
    
    # Identify all positive pairs to modify
    positive_pairs = []
    if entity_mode == 'diagnosis_only':
        for rel in relationships:
            for diag in rel.get('diagnoses', []):
                positive_pairs.append({'entity_label': diag['diagnosis'], 'date': rel['date']})
    else:  # multi_entity
        for rel in relationships:
            positive_pairs.append({'entity_label': rel.get('entity_label', ''), 'date': rel.get('date', '')})
    
    # Shuffle to apply changes in a random order
    random.shuffle(positive_pairs)
    
    # Insert text for a subset of pairs to avoid making the note too long too quickly
    for pair in positive_pairs[:2]:  # Modify up to 2 pairs per note
        entity_label = pair['entity_label']
        date_str = pair['date']
        
        if entity_label in entity_positions and date_str in date_positions:
            entity_start, entity_end = entity_positions[entity_label]
            date_start, date_end = date_positions[date_str]
            
            # Determine the space between the entity and the date
            start_point = min(entity_end, date_end)
            end_point = max(entity_start, date_start)
            
            if start_point < end_point:
                # Entity and date are not overlapping, insert between them
                insertion_point = random.randint(start_point, end_point)
            else:
                # Entity and date are adjacent or overlapping, insert after both
                insertion_point = max(entity_end, date_end)
            
            # Create a larger, multi-sentence filler text
            num_sentences = random.randint(2, 3)  # Choose to insert 2 or 3 sentences
            filler_sentences = random.choices(FILLER_TEXT, k=num_sentences)
            filler_text = " " + " ".join(filler_sentences) + " "
            
            # Insert the text
            note_text = insert_text_at_position(note_text, filler_text, insertion_point)
            
            # Update all positions after the insertion
            insert_length = len(filler_text)
            entities = recalculate_positions(entities, insertion_point, insert_length)
            dates = recalculate_positions(dates, insertion_point, insert_length)
            
            # Update the position lookups for the next iteration
            entity_positions = {e['label']: (e['start'], e['end']) for e in entities}
            date_positions = {d['parsed']: (d['start'], d['end']) for d in dates}
    
    return note_text, entities, dates

def insert_random_filler(note_text, min_length=20, max_length=100):
    """
    Insert random filler text into the note.
    
    Args:
        note_text: The original note text
        min_length: Minimum length of filler to insert
        max_length: Maximum length of filler to insert
    
    Returns:
        Tuple of (new_text, insert_position, insert_length)
    """
    # Choose a random filler text
    filler = random.choice(FILLER_TEXT)
    
    # Add a space or newline before the filler
    prefix = random.choice([" ", "\n"])
    filler = prefix + filler
    
    # Find a suitable position to insert the filler
    # Prefer inserting at the end of a sentence or line
    sentence_ends = [m.end() for m in re.finditer(r'\.(?=\s|$)', note_text)]
    line_ends = [m.end() for m in re.finditer(r'\n', note_text)]
    
    possible_positions = sentence_ends + line_ends
    
    if possible_positions:
        # Insert at a sentence or line end
        insert_position = random.choice(possible_positions)
    else:
        # Insert at a random position if no sentence/line ends found
        insert_position = random.randint(0, len(note_text))
    
    # Insert the filler text
    new_text = insert_text_at_position(note_text, filler, insert_position)
    
    return new_text, insert_position, len(filler)

# Create weighted distributions for entity selection
# The weights are inversely proportional to the index in the list (most common entities first)
def create_weighted_distribution(entities_list):
    """
    Create a weighted distribution for entity selection.
    Entities earlier in the list (more common) will have higher weights.
    
    Args:
        entities_list: List of entities
    
    Returns:
        List of weights for random.choices
    """
    n = len(entities_list)
    # Use a power law distribution (1/index) for weights
    # Add 1 to index to avoid division by zero
    weights = [1.0 / (i + 1) for i in range(n)]
    
    # Normalize weights to sum to 1.0
    total = sum(weights)
    normalized_weights = [w / total for w in weights]
    
    return normalized_weights

# Create weighted distributions once at module level
DIAGNOSIS_WEIGHTS = create_weighted_distribution(TOP_DIAGNOSIS_ENTITIES)
SNOMED_WEIGHTS = create_weighted_distribution(TOP_SNOMED_ENTITIES)
UMLS_WEIGHTS = create_weighted_distribution(TOP_UMLS_ENTITIES)

def insert_entity(note_text, entity_label, entity_category=None):
    """
    Insert an entity into the note text.
    
    Args:
        note_text: The original note text
        entity_label: The entity to insert
        entity_category: The category of the entity (for multi_entity mode)
    
    Returns:
        Tuple of (new_text, entity_dict) where entity_dict contains the entity info
    """
    # Find a suitable position to insert the entity
    # For simplicity, we'll insert at the end of a sentence
    sentence_ends = [m.end() for m in re.finditer(r'\.(?=\s|$)', note_text)]
    
    if sentence_ends:
        # Insert after a random sentence
        insert_position = random.choice(sentence_ends)
        prefix = " "
    else:
        # Insert at a random position if no sentence ends found
        insert_position = random.randint(0, len(note_text))
        prefix = " "
    
    # Create the phrase to insert
    phrases = [
        f"{prefix}Patient has a history of {entity_label}.",
        f"{prefix}Assessment includes {entity_label}.",
        f"{prefix}Noted {entity_label} in medical history.",
        f"{prefix}{entity_label} was discussed."
    ]
    insert_text = random.choice(phrases)
    
    # Insert the entity
    new_text = insert_text_at_position(note_text, insert_text, insert_position)
    
    # Calculate the position of the entity in the new text
    entity_start = insert_position + insert_text.find(entity_label)
    entity_end = entity_start + len(entity_label)
    
    # Create the entity dictionary
    if entity_category:
        # Multi-entity mode
        entity_dict = {
            "label": entity_label,
            "start": entity_start,
            "end": entity_end,
            "categories": [entity_category]
        }
    else:
        # Diagnosis-only mode
        entity_dict = {
            "label": entity_label,
            "start": entity_start,
            "end": entity_end
        }
    
    return new_text, entity_dict, insert_position, len(insert_text)

# ===== MAIN ENHANCEMENT FUNCTIONS =====

def enhance_diagnosis_only(row, target_length=None):
    """
    Enhance a single row of data in diagnosis_only mode.
    
    Args:
        row: A pandas Series representing a row in the DataFrame
        target_length: Target character length for the note
    
    Returns:
        Tuple of (updated_note, updated_extracted_disorders, updated_formatted_dates, updated_relationship_gold)
    """
    # Parse the JSON fields
    note_text = row['note']
    extracted_disorders = parse_json_field(row['extracted_disorders'])
    formatted_dates = parse_json_field(row['formatted_dates'])
    relationship_gold = parse_json_field(row['relationship_gold'])
    
    # First, reduce the number of dates to match the target
    target_dates = DIAGNOSIS_ONLY_TARGETS['avg_dates_per_document']
    formatted_dates, relationship_gold = reduce_date_density(formatted_dates, relationship_gold, target_dates)
    
    # Next, specifically increase distance for positive pairs
    note_text, extracted_disorders, formatted_dates = increase_positive_pair_distance(
        note_text, extracted_disorders, formatted_dates, relationship_gold, 'diagnosis_only'
    )
    
    # Set target length if not provided
    if target_length is None:
        current_length = len(note_text)
        target_length = max(DIAGNOSIS_ONLY_TARGETS['avg_document_length'], current_length)
    
    # Keep track of all modifications
    entity_position_map = {entity['label']: entity['start'] for entity in extracted_disorders}
    date_position_map = {date['parsed']: date['start'] for date in formatted_dates}
    
    # Enhance the note until it reaches the target length
    while len(note_text) < target_length:
        # Insert random filler text
        note_text, insert_position, insert_length = insert_random_filler(note_text)
        
        # Update entity positions
        extracted_disorders = recalculate_positions(extracted_disorders, insert_position, insert_length)
        formatted_dates = recalculate_positions(formatted_dates, insert_position, insert_length)
        
        # Update position maps
        entity_position_map = {entity['label']: entity['start'] for entity in extracted_disorders}
        date_position_map = {date['parsed']: date['start'] for date in formatted_dates}
    
    # Add additional entities based on target entity count
    current_entity_count = len(extracted_disorders)
    target_entity_count = DIAGNOSIS_ONLY_TARGETS['avg_entities_per_document']
    
    # Get existing entity labels to avoid duplicates
    existing_entity_labels = {entity['label'] for entity in extracted_disorders}
    
    while current_entity_count < target_entity_count:
        # Choose an entity using weighted distribution
        entity_label = random.choices(TOP_DIAGNOSIS_ENTITIES, weights=DIAGNOSIS_WEIGHTS, k=1)[0]
        
        # Skip if this entity already exists in the note
        if entity_label in existing_entity_labels:
            continue
        
        # Insert the entity
        note_text, entity_dict, insert_position, insert_length = insert_entity(note_text, entity_label)
        
        # Update entity positions
        extracted_disorders = recalculate_positions(extracted_disorders, insert_position, insert_length)
        formatted_dates = recalculate_positions(formatted_dates, insert_position, insert_length)
        
        # Add the new entity
        extracted_disorders.append(entity_dict)
        existing_entity_labels.add(entity_label)
        
        # Update position maps
        entity_position_map = {entity['label']: entity['start'] for entity in extracted_disorders}
        date_position_map = {date['parsed']: date['start'] for date in formatted_dates}
        
        current_entity_count += 1
    
    # Update relationship positions
    relationship_gold = update_relationship_positions(relationship_gold, entity_position_map, date_position_map)
    
    return note_text, extracted_disorders, formatted_dates, relationship_gold

def enhance_multi_entity(row, target_length=None):
    """
    Enhance a single row of data in multi_entity mode.
    
    Args:
        row: A pandas Series representing a row in the DataFrame
        target_length: Target character length for the note
    
    Returns:
        Tuple of (updated_note, updated_snomed, updated_umls, updated_formatted_dates, updated_relationship_gold)
    """
    # Parse the JSON fields
    note_text = row['note']
    snomed_entities = parse_json_field(row['extracted_snomed_entities'])
    umls_entities = parse_json_field(row['extracted_umls_entities'])
    formatted_dates = parse_json_field(row['formatted_dates'])
    relationship_gold = parse_json_field(row['relationship_gold'])
    
    # First, specifically increase distance for positive pairs
    note_text, snomed_entities, formatted_dates = increase_positive_pair_distance(
        note_text, snomed_entities, formatted_dates, relationship_gold, 'multi_entity'
    )
    # Also update UMLS entities positions to match SNOMED
    umls_entities = recalculate_positions(umls_entities, 0, 0)  # Just to ensure they're in sync
    
    # Set target length if not provided
    if target_length is None:
        current_length = len(note_text)
        target_length = max(MULTI_ENTITY_TARGETS['avg_document_length'], current_length)
    
    # --- RESTRUCTURED LOGIC ---
    # 1. ADD ENTITIES FIRST (New Priority)
    # This loop will run until the entity count target is met, regardless of note length
    current_entity_count = len(snomed_entities)
    target_entity_count = MULTI_ENTITY_TARGETS['avg_entities_per_document']
    existing_entity_labels = {entity['label'] for entity in snomed_entities}
    
    while current_entity_count < target_entity_count:
        if not TOP_SNOMED_ENTITIES: break # Safety break

        entity_index = random.choices(range(len(TOP_SNOMED_ENTITIES)), weights=SNOMED_WEIGHTS, k=1)[0]
        entity_label, entity_category = TOP_SNOMED_ENTITIES[entity_index]
        
        # Removed uniqueness check to allow duplicate entities (realistic in clinical notes)
        # and prevent infinite loops when target_entity_count > len(TOP_SNOMED_ENTITIES)
            
        # Insert the entity
        note_text, snomed_entity, insert_position, insert_length = insert_entity(
            note_text, entity_label, entity_category
        )
        
        # Get matching UMLS category
        if entity_index < len(TOP_UMLS_ENTITIES):
            _, umls_category = TOP_UMLS_ENTITIES[entity_index]
        else:
            umls_category = "Disease or Syndrome"
        
        umls_entity = deepcopy(snomed_entity)
        umls_entity['categories'] = [umls_category]
        
        # Update all positions
        snomed_entities = recalculate_positions(snomed_entities, insert_position, insert_length)
        umls_entities = recalculate_positions(umls_entities, insert_position, insert_length)
        formatted_dates = recalculate_positions(formatted_dates, insert_position, insert_length)
        
        # Add the new entities
        snomed_entities.append(snomed_entity)
        umls_entities.append(umls_entity)
        existing_entity_labels.add(entity_label)
        
        current_entity_count += 1

    # 2. ADD FILLER TEXT SECOND (If still needed)
    # This loop only runs if the note is still shorter than the target length after adding entities
    while len(note_text) < target_length:
        note_text, insert_position, insert_length = insert_random_filler(note_text)
        
        # Update entity positions
        snomed_entities = recalculate_positions(snomed_entities, insert_position, insert_length)
        umls_entities = recalculate_positions(umls_entities, insert_position, insert_length)
        formatted_dates = recalculate_positions(formatted_dates, insert_position, insert_length)
    
    # 3. UPDATE RELATIONSHIP POSITIONS (Final step)
    entity_position_map = {entity['label']: entity['start'] for entity in snomed_entities}
    date_position_map = {date['parsed']: date['start'] for date in formatted_dates}
    relationship_gold = update_relationship_positions(relationship_gold, entity_position_map, date_position_map)
    
    return note_text, snomed_entities, umls_entities, formatted_dates, relationship_gold

def enhance_synthetic_data(input_file, output_file, entity_mode='diagnosis_only'):
    """
    Enhance synthetic data to match real-world data statistics.
    
    Args:
        input_file: Path to the input CSV file
        output_file: Path to the output CSV file
        entity_mode: 'diagnosis_only' or 'multi_entity'
    """
    print(f"Enhancing {input_file} in {entity_mode} mode...")
    
    # Read the input CSV file
    df = pd.read_csv(input_file)
    
    # Create a copy for the enhanced data
    enhanced_df = df.copy()
    
    # Process each row
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Enhancing notes"):
        if entity_mode == 'diagnosis_only':
            # Enhance in diagnosis_only mode
            note_text, extracted_disorders, formatted_dates, relationship_gold = enhance_diagnosis_only(row)
            
            # Update the DataFrame
            enhanced_df.at[idx, 'note'] = note_text
            enhanced_df.at[idx, 'extracted_disorders'] = json.dumps(extracted_disorders)
            enhanced_df.at[idx, 'formatted_dates'] = json.dumps(formatted_dates)
            enhanced_df.at[idx, 'relationship_gold'] = json.dumps(relationship_gold)
            
        else:  # multi_entity mode
            # Enhance in multi_entity mode
            note_text, snomed_entities, umls_entities, formatted_dates, relationship_gold = enhance_multi_entity(row)
            
            # Update the DataFrame
            enhanced_df.at[idx, 'note'] = note_text
            enhanced_df.at[idx, 'extracted_snomed_entities'] = json.dumps(snomed_entities)
            enhanced_df.at[idx, 'extracted_umls_entities'] = json.dumps(umls_entities)
            enhanced_df.at[idx, 'formatted_dates'] = json.dumps(formatted_dates)
            enhanced_df.at[idx, 'relationship_gold'] = json.dumps(relationship_gold)
    
    # Save the enhanced data
    enhanced_df.to_csv(output_file, index=False)
    print(f"Enhanced data saved to {output_file}")

# ===== MAIN FUNCTION =====

def main():
    """Main function to run the enhancement script."""
    parser = argparse.ArgumentParser(description='Enhance synthetic data to match real-world statistics.')
    parser.add_argument('input_file', help='Path to the input CSV file')
    parser.add_argument('output_file', help='Path to the output CSV file')
    parser.add_argument('--mode', choices=['diagnosis_only', 'multi_entity'], default='diagnosis_only',
                        help='Entity mode to use for enhancement')
    
    args = parser.parse_args()
    
    # Validate that the top entities lists are populated
    if args.mode == 'diagnosis_only' and not TOP_DIAGNOSIS_ENTITIES:
        print("Error: TOP_DIAGNOSIS_ENTITIES list is empty. Please populate it with entities from the real dataset.")
        return
    
    if args.mode == 'multi_entity' and (not TOP_SNOMED_ENTITIES or not TOP_UMLS_ENTITIES):
        print("Error: TOP_SNOMED_ENTITIES or TOP_UMLS_ENTITIES list is empty. Please populate them with entities from the real dataset.")
        return
    
    # Enhance the data
    enhance_synthetic_data(args.input_file, args.output_file, args.mode)

if __name__ == "__main__":
    main()