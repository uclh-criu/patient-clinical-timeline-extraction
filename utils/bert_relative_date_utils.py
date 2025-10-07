from transformers import AutoTokenizer
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
import torch
import json
from general_utils import parse_jsonish

def create_token_label_dataset(df, tokenizer, label2id=None, max_length=384):
    if label2id is None:
        label2id = {"O": 0, "B-RELDATE": 1, "I-RELDATE": 2}

    examples = []

    for _, row in df.iterrows():
        text = row["note_text"]

        # --- FIX: parse JSON if it's a string ---
        rel_dates = row.get("relative_dates_json", [])
        if isinstance(rel_dates, str):
            try:
                rel_dates = json.loads(rel_dates)
            except Exception:
                rel_dates = parse_jsonish(rel_dates)  # fallback if available

        encoding = tokenizer(
            text,
            return_offsets_mapping=True,
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

        labels = ["O"] * len(encoding["offset_mapping"])

        for rd in rel_dates:
            if not isinstance(rd, dict):
                continue
            start, end = rd.get("start"), rd.get("end")
            if start is None or end is None:
                continue

            for i, (tok_start, tok_end) in enumerate(encoding["offset_mapping"]):
                if tok_end <= start or tok_start >= end:
                    continue
                if start <= tok_start < end:
                    if labels[i] == "O":
                        labels[i] = "B-RELDATE"
                        start = -1
                    else:
                        labels[i] = "I-RELDATE"

        encoding["labels"] = [label2id[l] for l in labels]
        encoding.pop("offset_mapping")
        examples.append(encoding)

    return examples

# Oversampling sequences with relative date tokens

def upsample_relative_date_sequences(dataset, factor=3):
    """
    Repeat samples that contain at least one relative date token (B or I)
    to balance the dataset.
    """
    positives = []
    negatives = []

    for ex in dataset:
        if any(l in [1, 2] for l in ex["labels"]):
            positives.append(ex)
        else:
            negatives.append(ex)

    print(f"Before upsampling: {len(positives)} with relative dates, {len(negatives)} without.")

    # Repeat positives
    positives_upsampled = positives * factor
    new_dataset = positives_upsampled + negatives

    # Shuffle to mix them
    from random import shuffle
    shuffle(new_dataset)

    print(f"After upsampling: {len(new_dataset)} total samples")
    return new_dataset


def compute_token_metrics(eval_pred, id2label=None):
    """
    Compute token-level precision, recall, and F1 for relative date tags.
    """
    logits, labels = eval_pred
    preds = logits.argmax(-1)

    # Flatten
    preds = preds.flatten()
    labels = labels.flatten()

    # Remove padding tokens (label == -100)
    mask = labels != -100
    preds = preds[mask]
    labels = labels[mask]

    # Collapse BIO to binary: RELDATE vs O
    rel_tag_ids = [1, 2]  # B-RELDATE, I-RELDATE
    preds_binary = [1 if p in rel_tag_ids else 0 for p in preds]
    labels_binary = [1 if l in rel_tag_ids else 0 for l in labels]

    precision = precision_score(labels_binary, preds_binary, zero_division=0)
    recall = recall_score(labels_binary, preds_binary, zero_division=0)
    f1 = f1_score(labels_binary, preds_binary, zero_division=0)

    return {"precision": precision, "recall": recall, "f1": f1}

def predict_relative_dates(note_text, model, tokenizer, id2label=None, max_length=384):
    """
    Predict relative date spans from a raw clinical note.
    """
    if id2label is None:
        id2label = {0: "O", 1: "B-RELDATE", 2: "I-RELDATE"}

    model.eval()
    inputs = tokenizer(note_text, return_tensors="pt", truncation=True,
                       padding=True, max_length=max_length,
                       return_offsets_mapping=True)

    with torch.no_grad():
        outputs = model(**{k: v for k, v in inputs.items() if k != "offset_mapping"})
        preds = outputs.logits.argmax(-1).squeeze().tolist()

    offsets = inputs["offset_mapping"].squeeze().tolist()
    labels = [id2label[p] for p in preds]

    # Reconstruct spans
    spans = []
    current = None
    for label, (start, end) in zip(labels, offsets):
        if label == "B-RELDATE":
            if current:
                spans.append(current)
            current = {"start": start, "end": end}
        elif label == "I-RELDATE" and current:
            current["end"] = end
        else:
            if current:
                spans.append(current)
                current = None
    if current:
        spans.append(current)

    # Convert spans back to text
    results = []
    for i, s in enumerate(spans):
        value = note_text[s["start"]:s["end"]]
        results.append({
            "id": i + 1,
            "value": value.strip(),
            "start": s["start"],
            "end": s["end"]
        })

    return results