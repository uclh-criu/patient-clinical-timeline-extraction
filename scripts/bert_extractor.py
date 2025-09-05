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

def preprocess_input(note_text, entity, date, window_size=100, entity_marker='[E]', date_marker='[D]'):
    """
    Prepare the input text for BERT by extracting a context window and marking entity/date.
    """
    context = get_context_window(note_text, entity['start'], date['start'], window_size)
    # Find entity and date positions in context
    entity_text = note_text[entity['start']:entity['end']]
    date_text = note_text[date['start']:date['end']]
    entity_offset = context.find(entity_text)
    date_offset = context.find(date_text)
    entity_span = (entity_offset, entity_offset + len(entity_text))
    date_span = (date_offset, date_offset + len(date_text))

    # Insert markers in reverse order (rightmost first)
    spans = sorted([('entity', *entity_span), ('date', *date_span)], key=lambda x: x[1], reverse=True)
    marked = context
    for label, start, end in spans:
        if label == 'entity':
            marked = marked[:start] + entity_marker + marked[start:end] + entity_marker + marked[end:]
        else:
            marked = marked[:start] + date_marker + marked[start:end] + date_marker + marked[end:]
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