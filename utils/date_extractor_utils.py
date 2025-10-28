import re
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import pandas as pd
import json
import dateparser
from dateparser.search import search_dates

def clean_value(v: str) -> str:
    """General cleaning wrapper for both absolute and relative comparisons."""
    if not isinstance(v, str):
        return ""
    v = v.lower().strip()
    v = re.sub(r'\s+', ' ', v)
    v = re.sub(r'[^a-z0-9\s]', '', v)
    return v.strip()

def extract_absolute_dates(text: str):
    if not text:
        return []

    date_pattern = r"""
    \b(
        (?:(?:\d{1,2}(?:st|nd|rd|th)?\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*)|(?:(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2}(?:st|nd|rd|th)?)),?\s+\d{4}
        |
        \d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}
        |
        \d{4}[/.-]\d{1,2}[/.-]\d{1,2}
        |
        (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}
        |
        (?:\d{1,2}[/.-]\d{4}|\d{4}[/.-]\d{1,2})
        |
        \d{1.2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+'\d{2}
        |
        (?:19|20)\d{2}
    )\b
    """
    
    absolute_dates = []

    try:
        matches = re.finditer(date_pattern, text, re.IGNORECASE | re.VERBOSE)
    except re.error as e:
        print(f"Regex error: {e}")
        return []
        
    found_spans = set()
    
    for m in matches:
        substring = m.group(0)
        span = (m.start(), m.end())
        
        # Validate using the default, non-strict dateparser.parse()
        parsed = dateparser.parse(substring)
        
        if parsed and span not in found_spans:
            absolute_dates.append({
                'id': f"abs_{len(absolute_dates) + 1}",
                'value': substring.strip(),
                'start': m.start(),
                'end': m.end()
            })
            found_spans.add(span)

    return absolute_dates

#Relative date regex patterns
RELATIVE_DATE_PATTERNS = [
    # --- Simple temporal keywords ---
    (r'\b(today|yesterday|tomorrow|currently|now|presently)\b', 'common'),
    (r'\b(this|last|next)\s+(morning|evening|night)\b', 'part_of_day'),

    # --- Standard relative time phrases ---
    (r'\b(last|this|next)\s+(week|month|year|day)\b', 'time_unit'),
    (r'\b(last|this|next)\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b', 'day_of_week'),

    # --- Numeric + time unit with modifier (e.g. "3 days ago") ---
    (r'\b(\d+|few|couple|several|some|a)\s+(days?|weeks?|months?|years?)\s+(ago|before|earlier|prior|from\s+now|later)\b', 'numeric_relative'),

    # --- Standalone numeric + unit (e.g. "6 months", "3 years") ---
    # Useful when used with context or inferred from clinical style
    (r'\b(\d+|few|couple|several|some|a)\s+(days?|weeks?|months?|years?)\b', 'numeric_simple'),

    # --- Hours-level ranges (e.g. "last 24 hours", "past 48 hrs") ---
    (r'\b(last|past|previous|preceding)\s+\d+\s*(hours?|hrs?)\b', 'short_time_window'),

    # --- 'past', 'over', 'within' style ranges ---
    (r'\b(past|over|within)\s+(few\s+|couple\s+|several\s+|some\s+|last\s+|next\s+)?(days?|weeks?|months?|years?)\b', 'past_future_range'),

    # --- 'start/end/early/late/middle of <period>' ---
    (r'\b(start|end|early|late|beginning|middle)\s+(of\s+)?(the\s+)?(day|week|month|year|quarter|20\d{2})\b', 'range_period'),

    # --- Clinical time references / history phrases ---
    (r'\b(\d+|few|couple|several|some)\s+(year|month|week|day)s?\s*(history|prior)\b', 'history_period'),

    # --- 'prior to <event>' phrases ---
    (r'\b(prior\s+to\s+(admission|presentation|surgery|assessment|procedure|event|discharge|consultation))\b', 'prior_to_event'),

    # --- 'preceding' or 'previous' period phrases ---
    (r'\b(preceding|preceeding|previous)\s+(day|days|week|month|year)s?\b', 'preceding_period'),
]

#Mormalisation helper function (only used in evaluation)
def normalise_relative(s: str) -> str:
    """
    Utility for evaluation/comparison only.
    Used to normalise relative date expressions so that
    semantically equivalent phrases (e.g. "last few months" vs "past 3 months")
    can be matched during testing.
    """
    s = s.lower().strip()
    s = re.sub(r'\s+', ' ', s)                 # collapse multiple spaces
    s = re.sub(r'[^a-z0-9\s]', '', s)          # remove punctuation

    # Standardise quantifiers
    s = re.sub(r'\b(few|couple|several|some|a|one)\b', 'some', s)

    # Standardise temporal markers
    s = re.sub(r'\b(last|previous|prior|preceding|preceeding)\b', 'past', s)
    s = re.sub(r'\b(today|now|currently|presently)\b', 'today', s)
    s = re.sub(r'\btomorrow\b', 'next day', s)
    s = re.sub(r'\byesterday\b', 'past day', s)

    # Normalise units (singular form for consistent mapping)
    s = re.sub(r'\bhrs?\b', 'hour', s)
    s = re.sub(r'\bmos?\b', 'month', s)
    s = re.sub(r'\byrs?\b', 'year', s)
    s = re.sub(r'\bdays?\b', 'day', s)
    s = re.sub(r'\bweeks?\b', 'week', s)
    s = re.sub(r'\bmonths?\b', 'month', s)
    s = re.sub(r'\byears?\b', 'year', s)

    # Simplify relative markers
    s = re.sub(r'\b(ago|before|earlier|previously)\b', 'past', s)
    s = re.sub(r'\b(from now|later|ahead)\b', 'future', s)

    # Remove stopwords that don’t alter semantics
    s = re.sub(r'\b(of|the|in|on)\b', '', s)
    s = re.sub(r'\s+', ' ', s).strip()

    return s

#Core extraction function
def extract_relative_dates(text: str) -> List[Dict[str, Any]]:
    """
    Extract relative date mentions from text using regex-based patterns,
    applying filters to reduce noise (e.g. ages, absolute dates).
    """
    if not text:
        return []

    relative_dates = []
    seen = set()

    # Replace common number words (useful for later absolute conversion)
    NUM_WORDS = {
        'one': '1', 'two': '2', 'three': '3', 'four': '4', 'five': '5',
        'six': '6', 'seven': '7', 'eight': '8', 'nine': '9', 'ten': '10'
    }
    for word, num in NUM_WORDS.items():
        text = re.sub(fr'\b{word}\b', num, text, flags=re.IGNORECASE)

    for pattern, pattern_type in RELATIVE_DATE_PATTERNS:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            val = match.group(0).strip()

            # --- Filters for clinical noise ---
            if re.match(r'^\d{2,3}\s*years?$', val.lower()):
                # Likely an age, not a relative date (e.g. "65 years")
                continue
            if re.search(r'[/\-\'’]', val):
                # Skip if looks like a partial absolute date (e.g. 11/2013)
                continue

            # Skip duplicates
            if val.lower() in seen:
                continue
            seen.add(val.lower())

            relative_dates.append({
                'id': f"rel_{len(relative_dates) + 1}",
                'value': val,
                'start': match.start(),
                'end': match.end(),
                'pattern_type': pattern_type
            })

    return relative_dates