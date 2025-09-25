import pandas as pd
import torch
from collections import Counter
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import numpy as np
from bert_extractor_utils import preprocess_input
from transformers import TrainingArguments, Trainer

def build_gold_lookup(gold_relations):
    return set((g["entity"], g["date"]) for g in gold_relations)

def get_label_for_pair(entity_value, date_value, gold_set):
    return "relation" if (entity_value, date_value) in gold_set else "no_relation"

def create_training_pairs(samples, max_length=256):
    all_samples = []
    
    for sample in samples:
        gold_set = build_gold_lookup(sample['relations_json'])
        
        # Process absolute dates
        for entity in sample['entities_list']:
            for date in sample['dates']:
                label_str = get_label_for_pair(entity['value'], date['value'], gold_set)
                label = 1 if label_str == 'relation' else 0
                
                processed = preprocess_input(sample['note_text'], entity, date)
                processed['label'] = label

                processed['patient_id'] = sample.get('patient_id', '')
                processed['note_id'] = sample.get('note_id', '')
                processed['distance'] = abs(entity['start'] - date['start'])
                
                all_samples.append(processed)
        
        # Process relative dates
        if 'relative_dates' in sample and sample['relative_dates']:
            for entity in sample['entities_list']:
                for rel_date in sample['relative_dates']:
                    label_str = get_label_for_pair(entity['value'], rel_date['value'], gold_set)
                    label = 1 if label_str == 'relation' else 0
                    
                    processed = preprocess_input(sample['note_text'], entity, rel_date)
                    processed['label'] = label

                    processed['patient_id'] = sample.get('patient_id', '')
                    processed['note_id'] = sample.get('note_id', '')
                    processed['distance'] = abs(entity['start'] - rel_date['start'])
                    
                    all_samples.append(processed)
    
    return pd.DataFrame(all_samples)

def compute_class_weights(dataset, num_labels):
    if len(dataset) == 0:
        return torch.ones(num_labels)
    
    counts = Counter([int(x) for x in dataset['label']])
    total = sum(counts.values())
    weights = []
    
    for i in range(num_labels):
        c = max(1, counts.get(i, 0))
        w = total / (num_labels * c)
        weights.append(w)
    
    # Normalize so mean weight ~= 1
    mean_w = sum(weights) / len(weights)
    weights = [w / mean_w for w in weights]
    return torch.tensor(weights, dtype=torch.float)

def downsample_classes(df, ratio=1.0, random_state=42):
    pos = df[df['label'] == 1]
    neg = df[df['label'] == 0]
    n_neg = int(len(pos) * ratio)
    neg_down = neg.sample(n=min(n_neg, len(neg)), random_state=random_state)
    return pd.concat([pos, neg_down]).sample(frac=1, random_state=random_state).reset_index(drop=True)

def upsample_classes(df, ratio=1.0, random_state=42):
    pos = df[df['label'] == 1]
    neg = df[df['label'] == 0]
    n_pos = int(len(neg) * ratio)
    pos_up = pos.sample(n=n_pos, replace=True, random_state=random_state)
    return pd.concat([pos_up, neg]).sample(frac=1, random_state=random_state).reset_index(drop=True)

def handle_class_imbalance(df, method='weighted', ratio=1.0, random_state=42):
    if method == 'weighted':
        return df, compute_class_weights(df, 2)
    elif method == 'downsample':
        balanced_df = downsample_classes(df, ratio, random_state)
        return balanced_df, None
    elif method == 'upsample':
        balanced_df = upsample_classes(df, ratio, random_state)
        return balanced_df, None
    else:
        raise ValueError("Method must be 'weighted', 'downsample', or 'upsample'")

def add_special_tokens(tokenizer):
    special_tokens = {'additional_special_tokens': ["[E1]", "[/E1]", "[E2]", "[/E2]"]}
    tokenizer.add_special_tokens(special_tokens)
    return tokenizer

def tokenize_function(example, tokenizer, max_length=256):
    return tokenizer(example["marked_text"], truncation=True, padding="max_length", max_length=max_length)

def run_sweep(model_init, training_args, param_grid, train_dataset, val_dataset, test_dataset, compute_metrics):
    """
    Runs hyperparameter sweep and collects results in a DataFrame.
    """
    results = []

    for i, params in enumerate(param_grid):
        args = TrainingArguments(
            output_dir=training_args.output_dir,
            eval_strategy=training_args.eval_strategy,
            save_strategy=training_args.save_strategy,
            load_best_model_at_end=True,
            logging_strategy=training_args.logging_strategy,
            logging_steps=training_args.logging_steps,
            metric_for_best_model=training_args.metric_for_best_model,
            greater_is_better=training_args.greater_is_better,
            num_train_epochs=params["epochs"],
            learning_rate=params["learning_rate"],
            per_device_train_batch_size=params["batch_size"],
            per_device_eval_batch_size=training_args.per_device_eval_batch_size,
            warmup_ratio=training_args.warmup_ratio,
            weight_decay=training_args.weight_decay,
            fp16=training_args.fp16,
            report_to=training_args.report_to,
            seed=training_args.seed,
        )

        trainer = Trainer(
            model=model_init(),
            args=args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
        )

        trainer.train()
        eval_results = trainer.evaluate(test_dataset)

        row = {
            "run": i,
            "batch_size": params["batch_size"],
            "learning_rate": params["learning_rate"],
            "epochs": params["epochs"],
            **eval_results
        }
        results.append(row)

    df = pd.DataFrame(results)
    best_idx = df["eval_positive_f1"].idxmax()
    df.loc[best_idx, "best"] = True
    return df

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    
    # Overall metrics
    metrics = {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
        "f1_micro": f1_score(labels, preds, average="micro", zero_division=0),
        "f1_weighted": f1_score(labels, preds, average="weighted", zero_division=0),
    }
    
    # Per-class metrics (focusing on positive class)
    pos_precision = precision_score(labels, preds, pos_label=1, zero_division=0)
    pos_recall = recall_score(labels, preds, pos_label=1, zero_division=0)
    pos_f1 = f1_score(labels, preds, pos_label=1, zero_division=0)
    
    metrics.update({
        "positive_precision": pos_precision,
        "positive_recall": pos_recall,
        "positive_f1": pos_f1
    })
    
    # Confusion matrix stats
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    metrics.update({
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp)
    })
    
    return metrics