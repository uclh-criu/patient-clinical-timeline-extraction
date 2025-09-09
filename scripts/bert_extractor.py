"""
BERT-based relation extraction utilities.

This module provides functions for preprocessing clinical text and performing
relation extraction using BERT models. It uses full-text context with distinct
entity markers for better relation extraction performance.
"""

def mark_entities_full_text(text, disorder_start, disorder_end, date_start, date_end, 
                           disorder_text, date_text):
    """
    Mark entities in full text with distinct markers for better entity type recognition.
    
    Args:
        text: Full clinical note text
        disorder_start, disorder_end: Character positions of disorder entity
        date_start, date_end: Character positions of date entity
        disorder_text, date_text: Text content of the entities
        
    Returns:
        Text with entities marked as [E1] disorder_text [/E1] and [E2] date_text [/E2]
    """
    marked = text
    # Insert markers in reverse order (rightmost first) to avoid position shifts
    for span, token1, token2, ent_text, span_end in sorted(
        [(disorder_start, "[E1]", "[/E1]", disorder_text, disorder_end),
         (date_start, "[E2]", "[/E2]", date_text, date_end)],
        reverse=True
    ):
        marked = marked[:span] + f"{token1} {ent_text} {token2}" + marked[span_end:]
    return marked


def preprocess_input(note_text, disorder, date):
    """
    Preprocess input using full text approach with position-based labeling.
    
    This is the main preprocessing function that:
    1. Extracts entity positions and text
    2. Marks entities with distinct markers
    3. Returns structured data for training/inference
    
    Args:
        note_text: Full clinical note text
        disorder: Disorder entity dict with 'start', 'end' keys
        date: Date entity dict with 'start', 'end', 'original' keys
        
    Returns:
        Dictionary with 'text', 'marked_text', entity positions, and metadata
    """
    disorder_start, disorder_end = disorder['start'], disorder['end']
    date_start = date.get('start', None)
    if date_start is None:
        date_start = note_text.find(date['original'])
    date_end = date_start + len(date['original'])
    
    disorder_text = note_text[disorder_start:disorder_end]
    date_text = note_text[date_start:date_end]
    
    marked_text = mark_entities_full_text(
        note_text, disorder_start, disorder_end, date_start, date_end,
        disorder_text, date_text
    )
    
    return {
        'text': note_text,
        'marked_text': marked_text,
        'ent1_start': disorder_start, 'ent1_end': disorder_end,
        'ent2_start': date_start, 'ent2_end': date_end
    }


def bert_extraction(note_text, disorder, date, model, tokenizer):
    """
    Predict if a disorder and date are related using BERT.
    
    This function performs end-to-end relation extraction:
    1. Preprocesses the text with entity markers
    2. Tokenizes the input
    3. Runs inference with the model
    4. Returns prediction and confidence
    
    Args:
        note_text: Full clinical note text
        disorder: Disorder entity dict
        date: Date entity dict
        model: Trained BERT model (BertRC or standard)
        tokenizer: Tokenizer with special tokens
        
    Returns:
        Tuple of (prediction, confidence) where prediction is 0 or 1
    """
    processed = preprocess_input(note_text, disorder, date)
    input_text = processed['marked_text']
    
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    outputs = model(**inputs)
    logits = outputs.logits
    prediction = logits.argmax().item()
    confidence = logits.softmax(dim=1).max().item()
    return prediction, confidence


# ============================================================================
# LEGACY FUNCTIONS (kept for backward compatibility)
# ============================================================================

def get_context_window(note_text, entity_start, date_start, window_size=100):
    """
    Extract a window of text around the entity and date positions.
    Legacy function - use preprocess_input for new code.
    """
    min_pos = min(entity_start, date_start)
    max_pos = max(entity_start, date_start)
    start = max(0, min_pos - window_size)
    end = min(len(note_text), max_pos + window_size)
    return note_text[start:end]


def mark_entity(text, entity, marker="[E]"):
    """
    Insert a marker around the entity in the text.
    Legacy function - use mark_entities_full_text for new code.
    """
    start, end = entity['start'], entity['end']
    return text[:start] + marker + text[start:end] + marker + text[end:]


def mark_date(text, date, marker="[D]"):
    """
    Insert a marker around the date in the text.
    Legacy function - use mark_entities_full_text for new code.
    """
    start, end = date['start'], date['end']
    return text[:start] + marker + text[start:end] + marker + text[end:]