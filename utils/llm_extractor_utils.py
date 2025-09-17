import json
import re
import os
from openai import OpenAI

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

def llm_extraction_openai(prompt, model):
    client = OpenAI()
    
    response = client.responses.create(
        model=model,
        input=prompt,
        temperature=0
    )
    
    return response.output_text.strip()

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

def make_multi_prompt(note_text, prompt_filename, entities_list=None, dates=None):
    """
    Create a prompt for extracting all entity-date relationships from a note.
    """
    utils_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(utils_dir, ".."))
    prompt_dir = os.path.join(project_root, "prompts")
    prompt_path = os.path.join(prompt_dir, prompt_filename)
    prompt_template = load_prompt_template(prompt_path)
    
    # Format entities and dates if provided
    entities_str = "\n".join([f"- {e['value']} (ID: {e['id']})" for e in entities_list]) if entities_list else "Extract all relevant medical conditions/findings"
    dates_str = "\n".join([f"- {d['value']} (ID: {d['id']})" for d in dates]) if dates else "Extract all relevant dates"
    
    prompt = prompt_template.format(
        note_text=note_text,
        entities=entities_str,
        dates=dates_str
    )
    
    return prompt

def llm_extraction_multi_openai(prompt, model):
    """
    Call OpenAI to extract all entity-date relationships at once.
    Returns JSON array of relationships.
    """
    client = OpenAI()
    
    response = client.responses.create(
        model=model,
        input=prompt,
        temperature=0
    )
    
    # Clean the response - remove markdown code block if present
    raw_text = response.output_text.strip()
    if raw_text.startswith('```'):
        # Remove first line (```json) and last line (```)
        raw_text = '\n'.join(raw_text.split('\n')[1:-1])
    
    # Parse the response as JSON
    try:
        relationships = json.loads(raw_text)
        return relationships
    except json.JSONDecodeError as e:
        print(f"Warning: Could not parse LLM response as JSON. Error: {str(e)}")
        return []