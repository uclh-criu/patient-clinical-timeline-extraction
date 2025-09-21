def mark_entities_full_text(text, entity_start, entity_end, date_start, date_end, 
                            entity_text, date_text):
    marked = text
    
    # Insert markers in reverse order (rightmost first) to avoid position shifts
    for span, token1, token2, ent_text, span_end in sorted(
        [
            (entity_start, "[E1]", "[/E1]", entity_text, entity_end),
            (date_start, "[E2]", "[/E2]", date_text, date_end),
        ],
        reverse=True
    ):
        marked = marked[:span] + f"{token1} {ent_text} {token2}" + marked[span_end:]
    
    return marked

def preprocess_input(note_text, entity, date):
    
    entity_start, entity_end = entity['start'], entity['end']
    date_start = date.get('start', None)
    
    if date_start is None:
        date_start = note_text.find(date['value'])
    date_end = date_start + len(date['value'])
    
    entity_text = note_text[entity_start:entity_end]
    date_text = note_text[date_start:date_end]

    marked_text = mark_entities_full_text(
        note_text,
        entity_start, entity_end,
        date_start, date_end,
        entity_text,
        date_text
    )

    return {
        'text': note_text,
        'marked_text': marked_text,
        'ent1_start': entity_start, 'ent1_end': entity_end,
        'ent2_start': date_start, 'ent2_end': date_end
    }

def bert_extraction(note_text, entity, date, model, tokenizer):
    processed = preprocess_input(note_text, entity, date)
    input_text = processed['marked_text']
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    outputs = model(**inputs)
    logits = outputs['logits']
    prediction = logits.argmax().item()
    confidence = logits.softmax(dim=1).max().item()
    return prediction, confidence