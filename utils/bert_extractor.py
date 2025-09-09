def mark_entities_full_text(text, disorder_start, disorder_end, date_start, date_end, 
                           disorder_text, date_text):
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
    processed = preprocess_input(note_text, disorder, date)
    input_text = processed['marked_text']
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    outputs = model(**inputs)
    logits = outputs.logits
    prediction = logits.argmax().item()
    confidence = logits.softmax(dim=1).max().item()
    return prediction, confidence