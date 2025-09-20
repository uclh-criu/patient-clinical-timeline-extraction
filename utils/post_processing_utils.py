import pandas as pd
import plotly.graph_objects as go
from dateutil.parser import parse
import re
from datetime import datetime
from date_extractor_utils import RELATIVE_DATE_PATTERNS, _calculate_absolute_date

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
        title=f"Clinical Timeline for Patient {patient_id}",
        xaxis_title="Date",
        yaxis_title="Clinical Events",
        yaxis=dict(
            autorange="reversed"
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
    timeline_summary = {
        'patient_id': patient_id,
        'total_events': len(patient_df),
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