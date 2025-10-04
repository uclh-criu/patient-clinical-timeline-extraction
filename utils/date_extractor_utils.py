import re
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import pandas as pd
import json
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
    """
    Extract cleaner absolute dates using dateparser.
    Returns date strings and positions.
    """
    if not text:
        return []

    absolute_dates = []
    # Regex to find plausible date substrings first (filters out noise)
    date_pattern = r'\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{2,4})\b'
    matches = list(re.finditer(date_pattern, text, re.IGNORECASE))

    for m in matches:
        substring = m.group(0)
        parsed = search_dates(substring)
        # Keep only if it's truly parseable as a date
        if parsed:
            absolute_dates.append({
                'id': f"abs_{len(absolute_dates) + 1}",
                'value': substring.strip(),
                'start': m.start(),
                'end': m.end()
            })

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

#Mormalisation helper function
def normalise_relative(s: str) -> str:
    """Clean and standardise relative date strings for consistent matching and mapping."""
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

#Apply to dataframe
def add_relative_dates(df):
    """
    Add relative_dates_json column to dataframe by extracting relative dates from text.
    """
    relative_dates_list = []
    for _, row in df.iterrows():
        relative_dates = extract_relative_dates(row.get('note_text', ''))
        relative_dates_list.append(relative_dates)

    df['relative_dates_json'] = [json.dumps(rd) for rd in relative_dates_list]
    return df

#Absolute date conversion
def _calculate_absolute_date(text: str, pattern_type: str, doc_date: datetime, groups: tuple) -> Optional[datetime]:
    """Convert relative expressions into absolute dates using the document timestamp."""
    text_lower = text.lower()

    try:
        if pattern_type == 'time_unit':
            return _handle_time_unit(text_lower, doc_date, groups)
        elif pattern_type == 'day_of_week':
            return _handle_day_of_week(text_lower, doc_date, groups)
        elif pattern_type == 'numeric_relative':
            return _handle_numeric_relative(text_lower, doc_date)
        elif pattern_type == 'common':
            return _handle_common(text_lower, doc_date)
        elif pattern_type == 'past_future_range':
            return _handle_range(text_lower, doc_date)
        elif pattern_type == 'range_period':
            return _handle_range_period(text_lower, doc_date)
    except Exception as e:
        print(f"[WARN] Error calculating date for '{text}': {e}")
        return None

    return None

#Individual handler functions
def _handle_time_unit(text: str, doc_date: datetime, groups: tuple) -> Optional[datetime]:
    """Handle 'last week', 'this month', 'next year'."""
    modifier, unit = groups
    unit = unit.lower()

    delta = {'day': 1, 'week': 7, 'month': 30, 'year': 365}.get(unit, 0)
    if modifier == 'last':
        return doc_date - timedelta(days=delta)
    elif modifier == 'next':
        return doc_date + timedelta(days=delta)
    else:  # this
        return doc_date
        
def _handle_day_of_week(text: str, doc_date: datetime, groups: tuple) -> Optional[datetime]:
    """Handle 'last Monday', 'next Friday'."""
    modifier, day_name = groups
    day_map = {
        'monday': 0, 'tuesday': 1, 'wednesday': 2, 'thursday': 3,
        'friday': 4, 'saturday': 5, 'sunday': 6
    }

    target = day_map.get(day_name.lower())
    if target is None:
        return None

    current = doc_date.weekday()
    diff = target - current

    if modifier == 'last':
        if diff >= 0: diff -= 7
    elif modifier == 'next':
        if diff <= 0: diff += 7

    return doc_date + timedelta(days=diff)

def _handle_numeric_relative(text: str, doc_date: datetime) -> Optional[datetime]:
    """Handle numeric expressions like '3 days ago' or '2 weeks later'."""
    m = re.search(r'(\d+)\s+(day|week|month|year)s?\s+(ago|before|earlier|prior)', text)
    if m:
        num, unit, _ = m.groups()
        delta = {'day': 1, 'week': 7, 'month': 30, 'year': 365}[unit]
        return doc_date - timedelta(days=int(num) * delta)

    m = re.search(r'(\d+)\s+(day|week|month|year)s?\s+(from now|later)', text)
    if m:
        num, unit, _ = m.groups()
        delta = {'day': 1, 'week': 7, 'month': 30, 'year': 365}[unit]
        return doc_date + timedelta(days=int(num) * delta)

    return None

def _handle_common(text: str, doc_date: datetime) -> Optional[datetime]:
    """Handle 'yesterday', 'today', 'tomorrow'."""
    if 'yesterday' in text:
        return doc_date - timedelta(days=1)
    elif 'today' in text:
        return doc_date
    elif 'tomorrow' in text:
        return doc_date + timedelta(days=1)
    return None

def _handle_range(text: str, doc_date: datetime) -> Optional[datetime]:
    """Handle 'past 2 weeks', 'within 3 months', etc."""
    m = re.search(r'(past|within|over)\s+(\d+)\s+(day|week|month|year)', text)
    if m:
        _, num, unit = m.groups()
        delta = {'day': 1, 'week': 7, 'month': 30, 'year': 365}[unit]
        return doc_date - timedelta(days=int(num) * delta)
    return None

def _handle_range_period(text: str, doc_date: datetime) -> Optional[datetime]:
    """Handle 'start of 2019', 'end of this month', etc."""
    if 'start' in text or 'beginning' in text:
        if 'year' in text:
            return doc_date.replace(month=1, day=1)
        elif 'month' in text:
            return doc_date.replace(day=1)
        elif 'week' in text:
            return doc_date - timedelta(days=doc_date.weekday())
    elif 'end' in text:
        if 'year' in text:
            return doc_date.replace(month=12, day=31)
        elif 'month' in text:
            next_month = doc_date.replace(day=28) + timedelta(days=4)
            return next_month.replace(day=1) - timedelta(days=1)
        elif 'week' in text:
            return doc_date + timedelta(days=(6 - doc_date.weekday()))
    return None