import re
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import pandas as pd
import json
import datefinder

def extract_absolute_dates(text: str) -> List[Dict[str, Any]]:
    """
    Extract absolute dates from text using datefinder.
    
    Args:
        text: Clinical note text
        
    Returns:
        List of absolute date dictionaries with original text and positions
    """
    if not text:
        return []
    
    absolute_dates = []
    
    for date_obj, source_str, indices in datefinder.find_dates(
        text, 
        source=True, 
        index=True, 
        strict=False
    ):
        absolute_dates.append({
            'id': f"abs_{len(absolute_dates) + 1}",
            'value': source_str,
            'start': indices[0],
            'end': indices[1]
        })
    
    return absolute_dates

# Patterns where we can confidently calculate dates
RELATIVE_DATE_PATTERNS = [
    # Time-based patterns (last/this/next + time unit)
    (r'\b(last|this|next)\s+(week|month|year|day)\b', 'time_unit'),
    (r'\b(last|this|next)\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b', 'day_of_week'),
    
    # Numeric + time unit patterns
    (r'\b(\d+)\s+(days?|weeks?|months?|years?)\s+ago\b', 'ago'),
    (r'\b(\d+)\s+(days?|weeks?|months?|years?)\s+from\s+now\b', 'from_now'),
    (r'\b(\d+)\s+(days?|weeks?|months?|years?)\s+earlier\b', 'earlier'),
    (r'\b(\d+)\s+(days?|weeks?|months?|years?)\s+later\b', 'later'),
    
    # Common relative phrases
    (r'\b(yesterday|today|tomorrow)\b', 'common'),
    (r'\b(earlier\s+this\s+(week|month|year))\b', 'earlier_period'),
    (r'\b(later\s+this\s+(week|month|year))\b', 'later_period'),
]

def add_relative_dates(df):
    """
    Add relative_dates_json column to dataframe by extracting relative dates from text.
    
    Args:
        df: Main dataframe with note_text column
        
    Returns:
        DataFrame with added relative_dates_json column
    """
    # Extract relative dates for each row
    relative_dates_list = []
    for _, row in df.iterrows():
        # Extract relative dates without document timestamp
        relative_dates = extract_relative_dates(row['note_text'])
        relative_dates_list.append(relative_dates)
    
    # Add as JSON column
    df['relative_dates_json'] = [json.dumps(rd) for rd in relative_dates_list]
    
    return df

def extract_relative_dates(text: str) -> List[Dict[str, Any]]:
    """
    Extract relative dates from text without calculating absolute dates.
    
    Args:
        text: Clinical note text
        
    Returns:
        List of relative date dictionaries with original text and positions
    """
    if not text:
        return []
    
    relative_dates = []
    
    for pattern, pattern_type in RELATIVE_DATE_PATTERNS:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            original_text = match.group(0)
            start_pos = match.start()
            end_pos = match.end()
            
            relative_dates.append({
                'id': f"rel_{len(relative_dates) + 1}",
                'value': original_text,
                'start': start_pos,
                'end': end_pos,
                'pattern_type': pattern_type
            })
    
    return relative_dates

def _calculate_absolute_date(text: str, pattern_type: str, doc_date: datetime, groups: tuple) -> Optional[datetime]:
    """Calculate absolute date from relative date text."""
    text_lower = text.lower()
    
    try:
        if pattern_type == 'time_unit':
            return _handle_time_unit(text_lower, doc_date, groups)
        elif pattern_type == 'day_of_week':
            return _handle_day_of_week(text_lower, doc_date, groups)
        elif pattern_type == 'ago':
            return _handle_ago(text_lower, doc_date, groups)
        elif pattern_type == 'from_now':
            return _handle_from_now(text_lower, doc_date, groups)
        elif pattern_type == 'earlier':
            return _handle_earlier(text_lower, doc_date, groups)
        elif pattern_type == 'later':
            return _handle_later(text_lower, doc_date, groups)
        elif pattern_type == 'common':
            return _handle_common(text_lower, doc_date)
        elif pattern_type == 'earlier_period':
            return _handle_earlier_period(text_lower, doc_date, groups)
        elif pattern_type == 'later_period':
            return _handle_later_period(text_lower, doc_date, groups)
    except Exception as e:
        print(f"Error calculating date for '{text}': {e}")
        return None
    
    return None

def _handle_time_unit(text: str, doc_date: datetime, groups: tuple) -> Optional[datetime]:
    """Handle patterns like 'last week', 'this month', 'next year'."""
    modifier, time_unit = groups
    
    if modifier == 'last':
        if time_unit == 'day':
            return doc_date - timedelta(days=1)
        elif time_unit == 'week':
            return doc_date - timedelta(weeks=1)
        elif time_unit == 'month':
            # Approximate month as 30 days
            return doc_date - timedelta(days=30)
        elif time_unit == 'year':
            return doc_date - timedelta(days=365)
    elif modifier == 'this':
        return doc_date
    elif modifier == 'next':
        if time_unit == 'day':
            return doc_date + timedelta(days=1)
        elif time_unit == 'week':
            return doc_date + timedelta(weeks=1)
        elif time_unit == 'month':
            return doc_date + timedelta(days=30)
        elif time_unit == 'year':
            return doc_date + timedelta(days=365)
    
    return None

def _handle_day_of_week(text: str, doc_date: datetime, groups: tuple) -> Optional[datetime]:
    """Handle patterns like 'last Monday', 'this Friday'."""
    modifier, day_name = groups
    day_map = {
        'monday': 0, 'tuesday': 1, 'wednesday': 2, 'thursday': 3,
        'friday': 4, 'saturday': 5, 'sunday': 6
    }
    
    target_day = day_map.get(day_name.lower())
    if target_day is None:
        return None
    
    current_weekday = doc_date.weekday()
    days_ahead = target_day - current_weekday
    
    if modifier == 'last':
        if days_ahead > 0:
            days_ahead -= 7
        return doc_date + timedelta(days=days_ahead)
    elif modifier == 'this':
        if days_ahead < 0:
            days_ahead += 7
        return doc_date + timedelta(days=days_ahead)
    elif modifier == 'next':
        if days_ahead <= 0:
            days_ahead += 7
        return doc_date + timedelta(days=days_ahead)
    
    return None

def _handle_ago(text: str, doc_date: datetime, groups: tuple) -> Optional[datetime]:
    """Handle patterns like '3 days ago', '2 weeks ago'."""
    number, time_unit = groups
    num = int(number)
    
    if 'day' in time_unit:
        return doc_date - timedelta(days=num)
    elif 'week' in time_unit:
        return doc_date - timedelta(weeks=num)
    elif 'month' in time_unit:
        return doc_date - timedelta(days=num * 30)
    elif 'year' in time_unit:
        return doc_date - timedelta(days=num * 365)
    
    return None

def _handle_from_now(text: str, doc_date: datetime, groups: tuple) -> Optional[datetime]:
    """Handle patterns like '3 days from now', '2 weeks from now'."""
    number, time_unit = groups
    num = int(number)
    
    if 'day' in time_unit:
        return doc_date + timedelta(days=num)
    elif 'week' in time_unit:
        return doc_date + timedelta(weeks=num)
    elif 'month' in time_unit:
        return doc_date + timedelta(days=num * 30)
    elif 'year' in time_unit:
        return doc_date + timedelta(days=num * 365)
    
    return None

def _handle_earlier(text: str, doc_date: datetime, groups: tuple) -> Optional[datetime]:
    """Handle patterns like '3 days earlier'."""
    return _handle_ago(text, doc_date, groups)

def _handle_later(text: str, doc_date: datetime, groups: tuple) -> Optional[datetime]:
    """Handle patterns like '3 days later'."""
    return _handle_from_now(text, doc_date, groups)

def _handle_common(text: str, doc_date: datetime) -> Optional[datetime]:
    """Handle common patterns like 'yesterday', 'today', 'tomorrow'."""
    if 'yesterday' in text:
        return doc_date - timedelta(days=1)
    elif 'today' in text:
        return doc_date
    elif 'tomorrow' in text:
        return doc_date + timedelta(days=1)
    
    return None

def _handle_earlier_period(text: str, doc_date: datetime, groups: tuple) -> Optional[datetime]:
    """Handle patterns like 'earlier this week'."""
    period = groups[0] if groups else 'week'
    
    if 'week' in period:
        # Start of current week (Monday)
        days_since_monday = doc_date.weekday()
        return doc_date - timedelta(days=days_since_monday)
    elif 'month' in period:
        # Start of current month
        return doc_date.replace(day=1)
    elif 'year' in period:
        # Start of current year
        return doc_date.replace(month=1, day=1)
    
    return None

def _handle_later_period(text: str, doc_date: datetime, groups: tuple) -> Optional[datetime]:
    """Handle patterns like 'later this week'."""
    period = groups[0] if groups else 'week'
    
    if 'week' in period:
        # End of current week (Sunday)
        days_until_sunday = 6 - doc_date.weekday()
        return doc_date + timedelta(days=days_until_sunday)
    elif 'month' in period:
        # End of current month (approximate)
        next_month = doc_date.replace(day=28) + timedelta(days=4)
        return next_month.replace(day=1) - timedelta(days=1)
    elif 'year' in period:
        # End of current year
        return doc_date.replace(month=12, day=31)
    
    return None