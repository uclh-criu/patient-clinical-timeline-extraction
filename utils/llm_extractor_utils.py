import json
import re
import os

def load_prompt_template(prompt_path):
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()

def make_binary_prompt(entity, date, note_text, prompt_filename):
    """
    Create a prompt for binary classification of entity-date relationship.
    Args:
        entity (dict): Entity info.
        date (dict): Date info.
        note_text (str): The clinical note.
        prompt_filename (str): Filename of the prompt template (e.g., "prompt1.txt").
    Returns:
        str: The full prompt for the model.
    """
    # Find the project root (assume utils/ is always one level below root)
    utils_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(utils_dir, ".."))
    prompt_dir = os.path.join(project_root, "prompts")
    prompt_path = os.path.join(prompt_dir, prompt_filename)
    prompt_template = load_prompt_template(prompt_path)

    prompt = (
        prompt_template + "\n"
        f"Entity: {entity['value']}\n"
        f"Date: {date['value']}\n"
        f"Note: {note_text}\n"
        "Answer:"
    )
    return prompt

def llm_extraction(prompt, generator, max_new_tokens=5):
    """
    Call the LLM with a binary prompt and return the full response.
    """
    outputs = generator(prompt, max_new_tokens=max_new_tokens, do_sample=False)
    return outputs[0]['generated_text']

def parse_llm_answer(full_response):
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