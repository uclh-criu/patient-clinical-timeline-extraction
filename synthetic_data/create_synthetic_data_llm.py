# synthetic_data/create_synthetic_data_llm.py
# Goal: Generate synthetic clinical notes via an LLM, guided by EDA stats from real data.
# - Loads a real CSV, runs minimal EDA to get target stats (length, entity/date counts, frequency).
# - Builds a precise prompt with instructions, examples, and EDA outputs.
# - Instructs the LLM to produce notes with machine-readable markup for:
#     - Entities (diagnosis, medication, procedure, symptom)
#     - Dates (multiple formats, including relative; include parsed ISO when possible)
#     - Explicit relationships between entity IDs and date IDs
#
# Markup conventions (easy to parse downstream):
#   - Entity:  <<ENT id=E1 type=diagnosis status=present experiencer=patient>>asthma<</ENT>>
#              <<ENT id=E2 type=medication status=present experiencer=patient>>metformin<</ENT>>
#              <<ENT id=E3 type=procedure status=present experiencer=patient>>MRI<</ENT>>
#              <<ENT id=E4 type=symptom status=negated experiencer=patient>>fever<</ENT>>
#   - Date:    <<DATE id=D1 original="(01 Jan 2024)" parsed="2024-01-01">>(01 Jan 2024)<</DATE>>
#              <<DATE id=D2 original="three months ago" parsed="2024-03-15">>three months ago<</DATE>>
#   - Relationships (end of note, machine-readable block):
#              [[RELATIONS]]
#              E1 -> D1
#              E2 -> D1
#              E3 -> D2
#              [[/RELATIONS]]
#
# Notes:
# - Some entities MUST be unlinked to any date (negated, other-experiencer, historical, etc.).
# - Dates should vary in format and include some relative dates (parsed using reference_date).
# - The EDA on a user-provided CSV guides length, entity/date counts, and entity variety.
# - OPENAI API key is read from .env; checks OPEN_API_KEY and OPENAI_API_KEY.

import os
import json
import argparse
from datetime import date
from typing import List, Dict, Any, Tuple

import pandas as pd
from dotenv import load_dotenv
import numpy as np

# EDA functions (local module in synthetic_data/)
import sys
from pathlib import Path
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.append(str(_THIS_DIR))

from eda_analysis import (
    get_doc_length_stats,
    get_entity_count_stats,
    get_entity_frequency,
    get_date_count_stats,
)

# Optional curated constants to improve variety/quality
from constants import FILLER_TEXT, TOP_DIAGNOSIS_ENTITIES

# You can use the new OpenAI SDK (if installed). Otherwise, swap to your LLM client of choice.
# pip install openai>=1.0.0
from openai import OpenAI


def make_json_safe(obj):
    """
    Recursively convert numpy/pandas types to native Python types so json.dumps works.
    """
    if isinstance(obj, dict):
        return {make_json_safe(k): make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_json_safe(x) for x in obj]
    if isinstance(obj, (np.integer, )):
        return int(obj)
    if isinstance(obj, (np.floating, )):
        return float(obj)
    if hasattr(obj, "item"):
        # catches some pandas/numpy scalars
        try:
            return obj.item()
        except Exception:
            return obj
    return obj


def load_eda_stats(
    csv_path: str,
    note_col: str = "note",
    entity_col: str = "extracted_disorders",
    date_col: str = "formatted_dates",
    backfill_entities: List[str] = None
) -> Dict[str, Any]:
    """
    Load a real dataset CSV and compute essential EDA stats to guide LLM generation.
    Returns a JSON-serializable dict of stats.
    """
    df = pd.read_csv(csv_path)
    eda = {
        "length_stats": get_doc_length_stats(df, note_col=note_col),
        "entity_count_stats": get_entity_count_stats(df, entity_col=entity_col),
        "date_count_stats": get_date_count_stats(df, date_col=date_col),
        "entity_frequency": get_entity_frequency(df, entity_col=entity_col),
        "columns_used": {"note_col": note_col, "entity_col": entity_col, "date_col": date_col},
        "rows": len(df),
        "source": os.path.basename(csv_path),
    }
    # Backfill curated entities to ensure breadth even if missing in data slice
    if backfill_entities:
        for e in backfill_entities:
            eda["entity_frequency"].setdefault(e, 1)

    return make_json_safe(eda)


def load_prompt_templates(path: str) -> Tuple[str, str]:
    """
    Load system and user prompt templates from a single prompt.txt file.
    The file must contain the two sections exactly as in prompt.txt:
      'SYSTEM PROMPT (paste into system_prompt):' ... then
      'USER PROMPT (paste into user_prompt):' ...
    Returns:
        (system_template, user_template)
    """
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    # Split on markers
    sys_marker = "SYSTEM PROMPT (paste into system_prompt):"
    usr_marker = "USER PROMPT (paste into user_prompt):"
    if sys_marker not in text or usr_marker not in text:
        raise ValueError("prompt.txt must include both system and user sections with the exact markers.")
    _, after_sys = text.split(sys_marker, 1)
    system_template, user_template = after_sys.split(usr_marker, 1)
    return system_template.strip(), user_template.strip()


def build_llm_prompt(
    eda_stats: Dict[str, Any],
    examples: List[str],
    reference_date: str,
    note_style_hint: str = "Outpatient clinic note with History/Exam/Plan sections",
    prompt_path: str = "synthetic_data/prompt.txt"
) -> Tuple[str, str]:
    """
    Build prompts using prompt.txt templates. Replaces placeholders:
    - {reference_date}, {note_style_hint}, {EDA_JSON}
    - Always appends any provided examples after the template's examples section.
    """
    system_template, user_template = load_prompt_templates(prompt_path)
    eda_json = json.dumps(make_json_safe(eda_stats), indent=2)

    # Render system prompt
    system_prompt = system_template.format(
        reference_date=reference_date,
        note_style_hint=note_style_hint
    )

    # Render user prompt and append extra examples (if any)
    user_prompt = user_template.replace("{EDA_JSON}", eda_json)

    if examples:
        examples_block = "\n\n--- ADDITIONAL EXAMPLES (names redacted; follow markup literally) ---\n\n"
        examples_block += "\n\n---\n".join(examples) + "\n\n--- END ADDITIONAL EXAMPLES ---"
        user_prompt = f"{user_prompt}\n{examples_block}"

    return system_prompt, user_prompt


def generate_llm_notes(
    csv_path: str,
    examples: List[str],
    reference_date: str = None,
    model: str = "gpt-4o",
    n_notes: int = 1,
    note_col: str = "note",
    entity_col: str = "extracted_disorders",
    date_col: str = "formatted_dates",
) -> List[str]:
    """
    Orchestrate EDA + prompt building + LLM call(s).
    Returns a list of generated notes (strings) including the [[RELATIONS]] block.
    """
    # Load EDA stats from real data
    eda_stats = load_eda_stats(
        csv_path=csv_path,
        note_col=note_col,
        entity_col=entity_col,
        date_col=date_col,
        backfill_entities=TOP_DIAGNOSIS_ENTITIES,  # optional curated backfill
    )

    # Default reference date is today if not provided
    if not reference_date:
        reference_date = str(date.today())

    # Build prompts
    system_prompt, user_prompt = build_llm_prompt(
        eda_stats=eda_stats,
        examples=examples,
        reference_date=reference_date,
        note_style_hint="Outpatient clinic note with History/Exam/Plan sections",
        prompt_path="synthetic_data/prompt.txt",  # <-- points to your file
    )

    # Read API key from .env (supports both OPEN_API_KEY and OPENAI_API_KEY)
    load_dotenv()
    api_key = os.getenv("OPEN_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPEN_API_KEY / OPENAI_API_KEY in .env")

    client = OpenAI(api_key=api_key)

    results = []
    for _ in range(n_notes):
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.8,
        )
        note_text = resp.choices[0].message.content
        results.append(note_text)

    return results


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic clinical notes via LLM, guided by EDA.")
    parser.add_argument("--csv", required=True, help="Path to CSV containing real data for EDA.")
    parser.add_argument("--n", type=int, default=1, help="Number of notes to generate.")
    parser.add_argument("--model", type=str, default="gpt-4o", help="LLM model name.")
    parser.add_argument("--examples_json", type=str, default="", help="Path to a JSON file containing a list of note examples (strings).")
    parser.add_argument("--reference_date", type=str, default="", help="Reference date (YYYY-MM-DD) for resolving relative dates. Defaults to today.")
    parser.add_argument("--note_col", type=str, default="note", help="Note column in CSV for EDA.")
    parser.add_argument("--entity_col", type=str, default="extracted_disorders", help="Entity column in CSV for EDA.")
    parser.add_argument("--date_col", type=str, default="formatted_dates", help="Date column in CSV for EDA.")
    args = parser.parse_args()

    examples = []
    if args.examples_json and os.path.isfile(args.examples_json):
        with open(args.examples_json, "r", encoding="utf-8") as f:
            examples = json.load(f)
            if not isinstance(examples, list):
                raise ValueError("examples_json must contain a JSON list of example note strings")

    notes = generate_llm_notes(
        csv_path=args.csv,
        examples=examples,
        reference_date=args.reference_date or None,
        model=args.model,
        n_notes=args.n,
        note_col=args.note_col,
        entity_col=args.entity_col,
        date_col=args.date_col,
    )

    # Print to stdout; in notebooks you can call generate_llm_notes directly and inspect results
    for i, n in enumerate(notes, 1):
        print(f"\n===== NOTE {i} =====\n")
        print(n)


if __name__ == "__main__":
    main()
