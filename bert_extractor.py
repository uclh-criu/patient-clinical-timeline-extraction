def bert_extraction(text, model, tokenizer):
    """Predict if a text describes a relationship (1) or not (0)"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    logits = outputs.logits
    prediction = logits.argmax().item()
    confidence = logits.softmax(dim=1).max().item()
    return prediction, confidence