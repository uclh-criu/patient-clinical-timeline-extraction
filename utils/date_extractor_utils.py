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

    # 'yesterday' and 'tomorrow'
    (r'\b(yesterday|tomorrow)\b', 'common_no_today'),
    (r'\b(this|last|next)\s+(morning|evening|night)\b', 'part_of_day'),

    # --- Standard relative time phrases ---
    (r'\b(last|this|next)\s+(week|month|year|day)\b', 'time_unit'),
    (r'\b(last|this|next)\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b', 'day_of_week'),

    # --- Numeric + time unit with modifier (e.g. "3 days ago") ---
    # UPDATED: Correctly handle singular units like "1 year ago"
    (r'\b([>~]?\d+(?:\.\d+)?|few|couple|several|some|a)\s+(day|week|month|year|yr|hour)s?\s+(ago|before|earlier|prior|post|after)\b', 'numeric_relative'),
    
    # --- This captures "last 2 weeks", "past 3 months", "last 24 hours" etc. ---
    # UPDATED: Generalised to better handle plural units like "last 9 months"
    (r'\b(last|past|previous|preceding)\s+([>~]?\d+(?:\.\d+)?|few|couple|several|some|a)\s+(day|week|month|year|yr|hour)s?\b', 'numeric_prefixed_range'),
    
    # --- 'past', 'over', 'within' style ranges (without numbers) ---
    (r'\b(past|over|within)\s+(few\s+|couple\s+|several\s+|some\s+|last\s+|next\s+)?(days?|weeks?|years?)\b', 'past_future_range'),

    # --- UPDATED: Generalised to handle more 'start/end of' phrases ---
    (r'\b(start|end|beginning|middle)\s+(of\s+)?(the\s+|this\s+|last\s+)?(day|week|month|year|quarter|20\d{2})\b', 'range_period'),

    # --- UPDATED: Made month patterns much stricter to reduce FPs ---
    (r'\b(since\s+(?:january|february|march|april|may|june|july|august|september|october|november|december)(?:\s+20\d{2})?)\b', 'since_month'),
    (r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+(last\s+year)\b', 'month_last_year'),

    # --- "since <year>" or "since <season> <year>" phrases ---
    (r'\b(since\s+(?:summer\s+|winter\s+|spring\s+|autumn\s+)?(?:19|20)\d{2})\b', 'since_year'),
    
    # --- UPDATED & CORRECTED: Handle numeric ranges like "2-3 years" or "6-8 weeks" ---
    (r'\b(last|past)\s+(\d+\s*-\s*\d+)\s+(days?|weeks?|months?|years?)\b', 'numeric_prefixed_range'),
    (r'\b(\d+\s*-\s*\d+)\s+(days?|weeks?|months?|years?)\s+(ago|prior|before|earlier)\b', 'numeric_range_modified'),
    
    # --- Clinical time references / history phrases ---
    (r'\b(\d+|few|couple|several|some)\s+(year|month|week|day)s?\s*(history|prior)\b', 'history_period'),

    # --- 'prior to <event>' phrases ---
    (r'\b(prior\s+to\s+(admission|presentation|surgery|assessment|procedure|event|discharge|consultation))\b', 'prior_to_event'),

    # --- 'preceding' or 'previous' period phrases ---
    (r'\b(preceding|previous)\s+(day|days|week|month|year)s?\b', 'preceding_period'),
]

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

    # --- UPDATED: More precise keywords for future dates ---
    FUTURE_KEYWORDS = ['next', 'tomorrow', 'later', 'from now', 'ahead', 'coming', 'within']

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
            val_lower = val.lower()

            # --- NEW: Filter out future dates ---
            if any(keyword in val_lower for keyword in FUTURE_KEYWORDS):
                continue

            # --- Filters for clinical noise ---
            if re.match(r'^\d{2,3}\s*years?$', val.lower()):
                # Likely an age, not a relative date (e.g. "65 years")
                continue
            
            # --- UPDATED: Correctly allow hyphens for all numeric range patterns ---
            if ('numeric_prefixed_range' not in pattern_type and 'numeric_range_modified' not in pattern_type) and re.search(r'[/\-\'’]', val):
                # Skip if it contains special chars AND is NOT a numeric range pattern
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

#Normalisation helper function (only used in evaluation)
def normalise_relative(s: str) -> str:
    """
    Utility for evaluation/comparison only.
    Used to normalise relative date expressions so that
    semantically equivalent phrases (e.g. "last few months" vs "past 3 months")
    can be matched during testing.
    """
    if not isinstance(s, str):
        return ""

    s = s.lower().strip()
    s = re.sub(r'\s+', ' ', s)                 # collapse multiple spaces
    s = re.sub(r'[^a-z0-9\s\-]', '', s)         # remove punctuation, but keep hyphens for ranges like 2-3

    # --- NEW: Convert number words to digits for consistency ---
    NUM_WORDS = {
        'one': '1', 'two': '2', 'three': '3', 'four': '4', 'five': '5',
        'six': '6', 'seven': '7', 'eight': '8', 'nine': '9', 'ten': '10'
    }
    for word, num in NUM_WORDS.items():
        s = re.sub(fr'\b{word}\b', num, s)

    # Standardise quantifiers (NOTE: 'one' is removed as it's now '1')
    s = re.sub(r'\b(few|couple|several|some|a)\b', 'some', s)

    # Standardise temporal markers
    s = re.sub(r'\b(last|previous|prior|preceding)\b', 'past', s)
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