import json
import re

def make_binary_prompt(entity, date, note_text, max_note_len=300):
    """
    Create a prompt for binary classification of entity-date relationship.
    """
    prompt = (
        "You are a clinical NLP assistant. For the following clinical note, "
        "determine if the entity and date are directly linked (i.e., the entity was diagnosed, treated, or mentioned as occurring on that date). "
        "Respond with only one word: Yes, No, or Unsure.\n\n"
        "Examples:\n"
        "Entity: asthma\nDate: 2024-08-02\nNote: Patient diagnosed with asthma on 2024-08-02.\nAnswer: Yes\n\n"
        "Entity: diabetes\nDate: 2024-08-02\nNote: Patient diagnosed with asthma on 2024-08-02. Patient also has diabetes.\nAnswer: No\n\n"
        "Entity: hypertension\nDate: 2024-08-02\nNote: Patient has hypertension, last reviewed in 2022.\nAnswer: No\n\n"
        "Entity: pneumonia\nDate: 2024-08-02\nNote: Patient may have pneumonia, last seen on 2024-08-02.\nAnswer: Unsure\n\n"
        f"Entity: {entity['label']}\n"
        f"Date: {date['parsed']}\n"
        f"Note: {note_text[:max_note_len]}\n"
        "Answer:"
    )
    return prompt

def llama_extraction(prompt, generator, max_new_tokens=5):
    """
    Call the Llama model with a binary prompt and return the full response.
    """
    outputs = generator(prompt, max_new_tokens=max_new_tokens, do_sample=False)
    return outputs[0]['generated_text']

def parse_llama_answer(full_response):
    """
    Extract the answer (Yes/No/Unsure) from the model's response.
    """
    response = full_response.strip().lower()
    if "yes" in response:
        return 1, 1.0
    elif "no" in response:
        return 0, 0.0
    else:
        return 0, 0.5  # Unsure or anything else