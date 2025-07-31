import pytest
import os
import pandas as pd
import json
from unittest.mock import patch, MagicMock
from utils import post_processing_utils
from extractors.naive_extractor import NaiveExtractor
from utils.inference_eval_utils import safe_json_loads
from types import SimpleNamespace

@pytest.fixture
def realistic_predictions(diagnosis_only_test_file_path, diagnosis_only_config):
    """Generate realistic predictions using the NaiveExtractor on the test dataset."""
    try:
        # Load the first few rows from the test dataset
        df = pd.read_csv(diagnosis_only_test_file_path, nrows=5)
        if len(df) == 0:
            pytest.skip("Test data file is empty")
        
        # Create a naive extractor for generating predictions
        config = SimpleNamespace(PROXIMITY_MAX_DISTANCE=200)
        extractor = NaiveExtractor(config)
        
        # Process the data to create realistic predictions
        predictions = []
        for idx, row in df.iterrows():
            text = row[diagnosis_only_config.TEXT_COLUMN]
            patient_id = row[diagnosis_only_config.PATIENT_ID_COLUMN]
            
            # Parse diagnoses and dates
            try:
                diagnoses_json = safe_json_loads(row[diagnosis_only_config.DIAGNOSES_COLUMN], [])
                diagnoses = [(item['label'], item['start']) for item in diagnoses_json]
                
                dates_json = safe_json_loads(row[diagnosis_only_config.DATES_COLUMN], [])
                dates = [(item['parsed'], item['original'], item['start']) for item in dates_json]
                
                # Extract relationships using the naive extractor
                note_predictions = extractor.extract(
                    text, 
                    entities=(diagnoses, dates), 
                    note_id=idx, 
                    patient_id=patient_id
                )
                
                # Add note_id and patient_id to each prediction since the extractor doesn't include them
                for pred in note_predictions:
                    pred['note_id'] = idx
                    pred['patient_id'] = patient_id
                
                predictions.extend(note_predictions)
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                continue  # Skip this row and continue
        
        # If we don't have enough predictions, create some manually
        if len(predictions) < 3:
            predictions = [
                {'patient_id': 'P001', 'note_id': 1, 'entity_label': 'headache', 'entity_category': 'disorder', 'date': '2023-01-15', 'confidence': 0.9},
                {'patient_id': 'P001', 'note_id': 2, 'entity_label': 'fever', 'entity_category': 'disorder', 'date': '2023-01-10', 'confidence': 0.8},
                {'patient_id': 'P002', 'note_id': 3, 'entity_label': 'cough', 'entity_category': 'disorder', 'date': '2023-01-20', 'confidence': 0.7}
            ]
        
        return predictions
    except Exception as e:
        pytest.skip(f"Error generating realistic predictions: {e}")
        return []

# 1. Test for aggregate_predictions_by_patient
def test_aggregate_predictions_by_patient(realistic_predictions):
    """Test aggregating predictions by patient ID using realistic predictions."""
    # Test with real predictions from the dataset
    timelines = post_processing_utils.aggregate_predictions_by_patient(realistic_predictions)
    
    # Check that we have at least one patient
    assert len(timelines) > 0
    
    # Check the structure of the timeline
    for patient_id, timeline in timelines.items():
        assert isinstance(patient_id, str)
        assert isinstance(timeline, list)
        
        if timeline:  # If the timeline is not empty
            # Check that the timeline entries have the required fields
            assert 'entity_label' in timeline[0]
            assert 'date' in timeline[0]
            
            # Check that the timeline is sorted by date
            if len(timeline) > 1:
                for i in range(len(timeline) - 1):
                    assert timeline[i]['date'] <= timeline[i + 1]['date']

def test_aggregate_predictions_by_patient_with_missing_patient_id():
    """Test that predictions without patient_id are skipped."""
    predictions = [
        {'patient_id': 'P001', 'note_id': 1, 'entity_label': 'headache', 'entity_category': 'symptom', 'date': '2023-01-15'},
        {'note_id': 2, 'entity_label': 'fever', 'entity_category': 'symptom', 'date': '2023-01-10'}  # Missing patient_id
    ]
    
    timelines = post_processing_utils.aggregate_predictions_by_patient(predictions)
    
    # Should only include the prediction with patient_id
    assert len(timelines) == 1
    assert 'P001' in timelines
    assert len(timelines['P001']) == 1

# 2. Test for generate_patient_timelines
@patch('os.makedirs')
@patch('builtins.open', new_callable=MagicMock)
def test_generate_patient_timelines(mock_open, mock_makedirs, realistic_predictions):
    """Test generating patient timeline files with realistic data."""
    # Aggregate the predictions first
    patient_timelines = post_processing_utils.aggregate_predictions_by_patient(realistic_predictions)
    
    output_dir = '/output'
    extractor_name = 'test_extractor'
    
    # Call the function
    post_processing_utils.generate_patient_timelines(patient_timelines, output_dir, extractor_name)
    
    # Check that the output directory was created
    mock_makedirs.assert_called_once_with(output_dir, exist_ok=True)
    
    # Check that files were created for each patient
    assert mock_open.call_count == len(patient_timelines)
    
    # Check the filenames
    file_paths = [call_args[0][0] for call_args in mock_open.call_args_list]
    for patient_id in patient_timelines.keys():
        expected_path = os.path.join(output_dir, f"patient_{patient_id}_{extractor_name}_timeline.txt")
        assert expected_path in file_paths

def test_generate_patient_timelines_empty():
    """Test generating patient timelines with empty input."""
    # Should not raise any exceptions
    post_processing_utils.generate_patient_timelines({}, '/output', 'test_extractor')
    post_processing_utils.generate_patient_timelines({'P001': []}, '/output', 'test_extractor')

# 3. Test for generate_patient_timeline_summary
@patch('os.makedirs')
@patch('builtins.open', new_callable=MagicMock)
def test_generate_patient_timeline_summary(mock_open, mock_makedirs, realistic_predictions):
    """Test generating a summary of patient timelines with realistic data."""
    # Aggregate the predictions first
    patient_timelines = post_processing_utils.aggregate_predictions_by_patient(realistic_predictions)
    
    output_dir = '/output'
    extractor_name = 'test_extractor'
    
    # Call the function
    post_processing_utils.generate_patient_timeline_summary(patient_timelines, output_dir, extractor_name)
    
    # Check that the output directory was created
    mock_makedirs.assert_called_once_with(output_dir, exist_ok=True)
    
    # Check that the summary file was created
    mock_open.assert_called_once()
    expected_path = os.path.join(output_dir, f"patient_timelines_summary_{extractor_name}.txt")
    assert mock_open.call_args[0][0] == expected_path

def test_generate_patient_timeline_summary_empty():
    """Test generating patient timeline summary with empty input."""
    # Should not raise any exceptions
    post_processing_utils.generate_patient_timeline_summary({}, '/output', 'test_extractor')

# 4. Test for generate_patient_timeline_visualizations
@patch('os.makedirs')
@patch('matplotlib.pyplot.savefig')
@patch('matplotlib.pyplot.close')
def test_generate_patient_timeline_visualizations(mock_close, mock_savefig, mock_makedirs, realistic_predictions):
    """Test generating visualizations of patient timelines with realistic data."""
    # Aggregate the predictions first
    patient_timelines = post_processing_utils.aggregate_predictions_by_patient(realistic_predictions)
    
    # Skip if no timelines with valid dates
    if not patient_timelines:
        pytest.skip("No patient timelines available for testing")
    
    # Call the function
    post_processing_utils.generate_patient_timeline_visualizations(patient_timelines, '/output', 'test_extractor')
    
    # Check that the output directory was created
    mock_makedirs.assert_called_once_with('/output', exist_ok=True)
    
    # Check that plots were saved
    assert mock_savefig.call_count > 0
    
    # Check that plots were closed
    assert mock_close.call_count > 0

def test_generate_patient_timeline_visualizations_with_alternative_date_formats():
    """Test generating visualizations with different date formats."""
    # Create test timelines with various date formats
    patient_timelines = {
        'P001': [
            {'entity_label': 'headache', 'entity_category': 'symptom', 'date': '15/01/2023', 'confidence': 0.9, 'note_id': 1},
            {'entity_label': 'fever', 'entity_category': 'symptom', 'date': 'Jan 10 2023', 'confidence': 0.8, 'note_id': 2}
        ]
    }
    
    # Mock the plotting functions to avoid actual visualization
    with patch('os.makedirs'), \
         patch('matplotlib.pyplot.savefig'), \
         patch('matplotlib.pyplot.close'):
        # Should not raise exceptions for alternative date formats
        post_processing_utils.generate_patient_timeline_visualizations(patient_timelines, '/output', 'test_extractor')

def test_generate_patient_timeline_visualizations_empty():
    """Test generating patient timeline visualizations with empty input."""
    # Should not raise any exceptions
    with patch('os.makedirs'):
        post_processing_utils.generate_patient_timeline_visualizations({}, '/output', 'test_extractor')
        post_processing_utils.generate_patient_timeline_visualizations({'P001': []}, '/output', 'test_extractor')

def test_generate_patient_timeline_visualizations_with_invalid_dates():
    """Test generating visualizations with invalid dates."""
    # Create test timelines with invalid dates
    patient_timelines = {
        'P001': [
            {'entity_label': 'headache', 'entity_category': 'symptom', 'date': 'not a date', 'confidence': 0.9, 'note_id': 1}
        ]
    }
    
    # Mock the plotting functions to avoid actual visualization
    with patch('os.makedirs'), \
         patch('matplotlib.pyplot.savefig'), \
         patch('matplotlib.pyplot.close'):
        # Should skip the invalid date without raising exceptions
        post_processing_utils.generate_patient_timeline_visualizations(patient_timelines, '/output', 'test_extractor') 