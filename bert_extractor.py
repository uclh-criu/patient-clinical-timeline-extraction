def get_context_window(note_text, entity_start, date_start, window_size=100):
    """
    Extract a window of text around the entity and date positions.
    Args:
        note_text (str): The full clinical note.
        entity_start (int): Start index of the entity in the note.
        date_start (int): Start index of the date in the note.
        window_size (int): Number of characters to include before and after the min/max position.
    Returns:
        str: The extracted context window.
    """
    min_pos = min(entity_start, date_start)
    max_pos = max(entity_start, date_start)
    start = max(0, min_pos - window_size)
    end = min(len(note_text), max_pos + window_size)
    return note_text[start:end]

def mark_entity(text, entity, marker="[E]"):
    """
    Insert a marker around the entity in the text.
    Args:
        text (str): The context window.
        entity (dict): Entity with 'start' and 'end' positions.
        marker (str): Marker to use for the entity.
    Returns:
        str: Text with the entity marked.
    """
    start, end = entity['start'], entity['end']
    return text[:start] + marker + text[start:end] + marker + text[end:]

def mark_date(text, date, marker="[D]"):
    """
    Insert a marker around the date in the text.
    Args:
        text (str): The context window.
        date (dict): Date with 'start' and 'end' positions.
        marker (str): Marker to use for the date.
    Returns:
        str: Text with the date marked.
    """
    start, end = date['start'], date['end']
    return text[:start] + marker + text[start:end] + marker + text[end:]

def preprocess_input(note_text, entity, date, window_size=100):
    """
    Prepare the input text for BERT by extracting a context window and marking entity/date.
    Args:
        note_text (str): The full clinical note.
        entity (dict): Entity with 'start' and 'end'.
        date (dict): Date with 'start' and 'end'.
        window_size (int): Context window size.
    Returns:
        str: Preprocessed input text for BERT.
    """
    context = get_context_window(note_text, entity['start'], date['start'], window_size)
    # Adjust entity/date positions relative to context window
    offset = context.find(note_text[entity['start']:entity['end']])
    entity_rel = {'start': offset, 'end': offset + (entity['end'] - entity['start'])}
    offset_date = context.find(note_text[date['start']:date['end']])
    date_rel = {'start': offset_date, 'end': offset_date + (date['end'] - date['start'])}
    marked = mark_entity(context, entity_rel)
    marked = mark_date(marked, date_rel)
    return marked

def bert_extraction(note_text, entity, date, model, tokenizer, window_size=100):
    """
    Predict if an entity and date are related using BERT.
    Args:
        note_text (str): The full clinical note.
        entity (dict): Entity with 'start', 'end', 'label'.
        date (dict): Date with 'start', 'end', 'parsed'.
        model: HuggingFace model.
        tokenizer: HuggingFace tokenizer.
        window_size (int): Context window size.
    Returns:
        int: Prediction (0 or 1).
        float: Confidence score.
    """
    input_text = preprocess_input(note_text, entity, date, window_size)
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    logits = outputs.logits
    prediction = logits.argmax().item()
    confidence = logits.softmax(dim=1).max().item()
    return prediction, confidence