import pandas as pd
import plotly.graph_objects as go
from dateutil.parser import parse
from dateutil.relativedelta import relativedelta
import re
from typing import Optional, Tuple
from datetime import datetime
from word2number import w2n

from date_extractor_utils import RELATIVE_DATE_PATTERNS

# Helper function to parse numbers
def _parse_number(text: str) -> Optional[float]:
    """
    Parses a number from a string, handling ranges, digits, decimals, and number-words.
    """
    text = text.lower()
    
    # Handle numeric ranges like "2-3", returning the larger number.
    range_match = re.search(r'(\d+)\s*-\s*(\d+)', text)
    if range_match:
        return float(range_match.group(2))

    # Handle qualitative terms like "a few", "a couple" BEFORE word-to-number
    if 'few' in text or 'couple' in text:
        return 2.0 # Approximation

    # Convert number words to numeric form
    try:
        # Handle 'a' or 'an' as 1, but be specific
        text = re.sub(r'\b(a|an)\b(?=\s+(day|week|month|year))', '1', text)
        processed_words = []
        words = text.split()
        # Process word by word to avoid w2n failing on non-number words
        for word in words:
            try:
                # Check if it's a number word that w2n knows
                if word in w2n.american_number_system:
                    processed_words.append(str(w2n.word_to_num(word)))
                else:
                    processed_words.append(word)
            except ValueError:
                processed_words.append(word)
        text = ' '.join(processed_words)
    except Exception:
        pass # Ignore if conversion fails

    # Find the first remaining digit or decimal number
    match = re.search(r'(\d+(?:\.\d+)?)', text)
    if match:
        return float(match.group(1))
        
    return None

#Individual handler functions
def _handle_time_unit(text: str, doc_date: datetime, groups: tuple) -> Optional[datetime]:
    """Handle 'last week', 'this month', 'next year'."""
    modifier, unit = groups
    unit = unit.lower().rstrip('s') # Remove plural 's'

    delta_map = {
        'day': relativedelta(days=1),
        'week': relativedelta(weeks=1),
        'month': relativedelta(months=1),
        'year': relativedelta(years=1)
    }
    delta = delta_map.get(unit)
    if not delta:
        return None

    if modifier == 'last':
        return doc_date - delta
    elif modifier == 'next':
        return doc_date + delta
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

    return doc_date + relativedelta(days=diff)

def _handle_numeric_relative(text: str, doc_date: datetime) -> Optional[datetime]:
    """
    Handle numeric expressions like '3 days ago', 'two weeks later', '1.5 years prior'.
    Uses the _parse_number helper and relativedelta for accuracy.
    """
    num = _parse_number(text)
    if num is None:
        return None

    # Determine the time unit
    unit_match = re.search(r'(day|week|month|year|yr|hour)s?', text, re.IGNORECASE)
    if not unit_match:
        return None
    unit = unit_match.group(1).lower().rstrip('s')

    # Create the relativedelta object
    if unit in ['year', 'yr']:
        # Handle decimal years by converting to months
        if isinstance(num, float) and not num.is_integer():
            total_months = int(num * 12)
            delta = relativedelta(months=total_months)
        else:
            delta = relativedelta(years=int(num))
    elif unit == 'month':
        delta = relativedelta(months=int(num))
    elif unit == 'week':
        delta = relativedelta(weeks=int(num))
    elif unit in ['day', 'hour']:
        delta = relativedelta(days=int(num)) # Treat hours as days for simplicity on timeline
    else:
        return None

    # Check for past or future keywords to determine direction
    if any(kw in text for kw in ['ago', 'before', 'earlier', 'prior', 'past']):
        return doc_date - delta
    elif any(kw in text for kw in ['from now', 'later', 'after', 'post']):
        return doc_date + delta
    
    # Default to past if no clear modifier (e.g., from 'numeric_prefixed_range' like 'last 2 weeks')
    return doc_date - delta


def _handle_common(text: str, doc_date: datetime) -> Optional[datetime]:
    """Handle 'yesterday', 'today', 'tomorrow'."""
    if 'yesterday' in text:
        return doc_date - relativedelta(days=1)
    elif 'today' in text:
        return doc_date
    elif 'tomorrow' in text:
        return doc_date + relativedelta(days=1)
    return None

def _handle_range(text: str, doc_date: datetime) -> Optional[datetime]:
    """Handle 'past 2 weeks', 'within 3 months', etc."""
    m = re.search(r'(past|within|over)\s+(\d+)\s+(day|week|month|year)', text)
    if m:
        _, num, unit = m.groups()
        delta = {'day': 1, 'week': 7, 'month': 30, 'year': 365}[unit]
        return doc_date - relativedelta(days=int(num) * delta)
    return None

def _handle_range_period(text: str, doc_date: datetime) -> Optional[datetime]:
    """Handle 'start of 2019', 'end of this month', etc."""
    if 'start' in text or 'beginning' in text:
        if 'year' in text:
            return doc_date.replace(month=1, day=1)
        elif 'month' in text:
            return doc_date.replace(day=1)
        elif 'week' in text:
            return doc_date - relativedelta(days=doc_date.weekday())
    elif 'end' in text:
        if 'year' in text:
            return doc_date.replace(month=12, day=31)
        elif 'month' in text:
            next_month = doc_date.replace(day=28) + relativedelta(days=4)
            return next_month.replace(day=1) - relativedelta(days=1)
        elif 'week' in text:
            return doc_date + relativedelta(days=(6 - doc_date.weekday()))
    return None

def _handle_since_year(text: str, doc_date: datetime) -> Optional[datetime]:
    """Handle 'since 2019', 'since summer 2020', etc."""
    match = re.search(r'(19|20)\d{2}', text)
    if match:
        year = int(match.group(0))
        return datetime(year, 1, 1)
    return None

def _handle_since_month(text: str, doc_date: datetime) -> Optional[datetime]:
    """Handle 'since January 2023' or 'since November'."""
    try:
        parsed_date = parse(text.replace('since', '').strip())
        if str(doc_date.year) not in text and str(doc_date.year-1) not in text:
            if parsed_date.month > doc_date.month:
                return parsed_date.replace(year=doc_date.year - 1, day=1)
            else:
                return parsed_date.replace(year=doc_date.year, day=1)
        return parsed_date.replace(day=1)
    except (ValueError, TypeError):
        return None

def _handle_prior_to_event(text: str, doc_date: datetime) -> Optional[datetime]:
    """Handle 'prior to admission'. Returns the document date as a proxy."""
    return doc_date

def _handle_part_of_day(text: str, doc_date: datetime, groups: tuple) -> Optional[datetime]:
    """Handle 'last evening', 'this morning', etc."""
    modifier, _ = groups
    if modifier == 'last':
        return doc_date - relativedelta(days=1)
    elif modifier == 'next':
        return doc_date + relativedelta(days=1)
    else: # 'this'
        return doc_date

def _handle_month_last_year(text: str, doc_date: datetime) -> Optional[datetime]:
    """Handle 'September last year'."""
    try:
        # Manually construct the date for accuracy
        parsed_month = parse(text, fuzzy=True)
        return datetime(doc_date.year, parsed_month.month, 1) - relativedelta(years=1)
    except (ValueError, TypeError):
        return None

def _handle_preceding_period(text: str, doc_date: datetime) -> Optional[datetime]:
    """Handle 'preceding days', 'previous week'."""
    if 'day' in text:
        return doc_date - relativedelta(days=1)
    elif 'week' in text:
        return doc_date - relativedelta(weeks=1)
    elif 'month' in text:
        return doc_date - relativedelta(months=1)
    elif 'year' in text:
        return doc_date - relativedelta(years=1)
    return None

#Absolute date conversion
def _calculate_absolute_date(text: str, pattern_type: str, doc_date: datetime, groups: tuple) -> Optional[datetime]:
    """Convert relative expressions into absolute dates using the document timestamp."""
    text_lower = text.lower()

    try:
        if pattern_type == 'time_unit':
            return _handle_time_unit(text_lower, doc_date, groups)
        elif pattern_type == 'day_of_week':
            return _handle_day_of_week(text_lower, doc_date, groups)
        elif pattern_type in ['numeric_relative', 'numeric_prefixed_range', 'history_period', 'numeric_range_modified']:
            return _handle_numeric_relative(text_lower, doc_date)
        elif pattern_type == 'common_no_today':
            return _handle_common(text_lower, doc_date)
        elif pattern_type == 'past_future_range':
            return _handle_range(text_lower, doc_date)
        elif pattern_type == 'range_period':
            return _handle_range_period(text_lower, doc_date)
        elif pattern_type == 'since_year':
            return _handle_since_year(text_lower, doc_date)
        elif pattern_type == 'since_month':
            return _handle_since_month(text_lower, doc_date)
        elif pattern_type == 'prior_to_event':
            return _handle_prior_to_event(text_lower, doc_date)
        elif pattern_type == 'part_of_day':
            return _handle_part_of_day(text_lower, doc_date, groups)
        elif pattern_type == 'month_last_year':
            return _handle_month_last_year(text_lower, doc_date)
        elif pattern_type == 'preceding_period':
            return _handle_preceding_period(text_lower, doc_date)
    except Exception as e:
        print(f"[WARN] Error calculating date for '{text}': {e}")
        return None

    return None

def standardize_date(row):
    """
    Parse a date string into a datetime object, handling both absolute and relative dates.
    """
    date_str = row['date']
    date_type = row.get('date_type', 'absolute') # Default to 'absolute' if not present
    doc_timestamp = row.get('document_timestamp')

    if not isinstance(date_str, str):
        return pd.NaT

    # Handle relative dates using existing logic
    if date_type == 'relative' and doc_timestamp:
        doc_date = pd.to_datetime(doc_timestamp)
        for pattern, pattern_type in RELATIVE_DATE_PATTERNS:
            match = re.search(pattern, date_str, re.IGNORECASE)
            if match:
                # Call the calculation function from the other utility
                return _calculate_absolute_date(date_str, pattern_type, doc_date, match.groups())
        return pd.NaT # Return NaT if no relative pattern matched

    # Handle absolute dates
    try:
        return parse(date_str, fuzzy=False)
    except (ValueError, TypeError):
        return pd.NaT

def create_interactive_patient_timeline(timeline_df, patient_id):
    """
    Create a timeline visualization for a specific patient.
    
    Args:
        timeline_df: DataFrame containing all patient timeline data
        patient_id: The specific patient_id to plot
    Returns:
        Plotly figure object or None if no data found
    """
    # Filter for specific patient
    patient_df = timeline_df[timeline_df['patient_id'] == patient_id]
    
    if patient_df.empty:
        print(f"No data found for patient {patient_id}")
        return None
        
    # Drop rows where date standardization might have failed and sort
    patient_df = patient_df.dropna(subset=['standardized_date']).sort_values(by='standardized_date')

    if patient_df.empty:
        print(f"No valid dates found for patient {patient_id}")
        return None
    
    # Calculate summary information
    unique_entities = patient_df['entity_preferred_name'].unique()
    total_entities = len(patient_df)  # Total number of entities/events
    date_range_start = patient_df['standardized_date'].min().strftime('%Y-%m-%d')
    date_range_end = patient_df['standardized_date'].max().strftime('%Y-%m-%d')
    
    height = max(600, len(unique_entities) * 30)
        
    fig = go.Figure(data=[
        go.Scatter(
            x=patient_df['standardized_date'],
            y=patient_df['entity_preferred_name'],
            mode='markers',
            marker=dict(size=10),
            hovertext=patient_df.apply(
                lambda row: f"<b>{row['entity_preferred_name']}</b><br>Date: {row['standardized_date'].strftime('%Y-%m-%d')}", 
                axis=1
            ),
            hoverinfo='text'
        )
    ])
    
    fig.update_layout(
        title=f"Clinical Timeline for Patient {patient_id}<br><sup>Date Range: {date_range_start} to {date_range_end} | {total_entities} Clinical Entities</sup>",
        xaxis_title="Date",
        yaxis_title="Clinical Events",
        height=height,
        width=1000,
        yaxis=dict(
            autorange="reversed",
            tickmode='array',
            ticktext=sorted(unique_entities),
            tickvals=sorted(unique_entities)
        ),
        margin=dict(
            l=200,
            r=20,
            t=80,
            b=50
        )
    )
    
    return fig

def plot_all_interactive_patient_timelines(timeline_df):
    """
    Create timeline visualizations for all patients in the dataset.
    
    Args:
        timeline_df: DataFrame containing all patient timeline data
    Returns:
        List of Plotly figure objects
    """
    figures = []
    
    # Group by patient_id and create timeline for each
    for patient_id in timeline_df['patient_id'].unique():
        fig = create_interactive_patient_timeline(timeline_df, patient_id)
        if fig:
            figures.append(fig)
    
    print(f"Created {len(figures)} timeline plots")
    return figures

def get_patient_timeline_summary(timeline_df, patient_id):
    """
    Create a structured summary of a patient's timeline.
    
    Args:
        timeline_df: DataFrame containing all patient timeline data
        patient_id: The specific patient_id to summarize
    Returns:
        dict: A structured summary of the patient's timeline
    """
    # Filter for specific patient
    patient_df = timeline_df[timeline_df['patient_id'] == patient_id].copy()
    
    if patient_df.empty:
        print(f"No data found for patient {patient_id}")
        return None
    
    # Drop rows where date standardization might have failed and sort
    patient_df = patient_df.dropna(subset=['standardized_date']).sort_values(by='standardized_date')
    
    # Create the timeline summary
    # Handle both numeric and string patient IDs for JSON serialization
    try:
        # Try to convert to int if it's numeric
        patient_id_serializable = int(patient_id)
    except (ValueError, TypeError):
        # If conversion fails, keep as string
        patient_id_serializable = str(patient_id)
    
    timeline_summary = {
        'patient_id': patient_id_serializable,
        'total_entities': len(patient_df),
        'date_range': {
            'start': patient_df['standardized_date'].min().strftime('%Y-%m-%d'),
            'end': patient_df['standardized_date'].max().strftime('%Y-%m-%d')
        },
        'events': []
    }
    
    # Add each event in chronological order
    for _, row in patient_df.iterrows():
        event = {
            'original_date': row['date'],
            'standardized_date': row['standardized_date'].strftime('%Y-%m-%d'),
            'date_type': row['date_type'],
            'entity_cui': row['entity_cui'],
            'entity_label': row['entity_label'],
            'entity_preferred_name': row['entity_preferred_name']
        }
        timeline_summary['events'].append(event)
    
    return timeline_summary