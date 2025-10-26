import pandas as pd
import plotly.graph_objects as go
from dateutil.parser import parse
import re
from typing import Optional
from datetime import datetime, timedelta
from date_extractor_utils import RELATIVE_DATE_PATTERNS

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
    
    # Calculate height based on number of unique entities (for spacing)
    height = max(600, len(unique_entities) * 30)  # minimum 600px, 30px per unique entity
        
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
            l=200,  # Left margin for labels
            r=20,
            t=80,  # Increased top margin to move title outside
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
            'date': row['standardized_date'].strftime('%Y-%m-%d'),
            'event': row['entity_preferred_name'],
            'date_type': row['date_type'],
            'original_date': row['date']
        }
        timeline_summary['events'].append(event)
    
    return timeline_summary