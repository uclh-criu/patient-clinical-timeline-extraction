import pytest
import json
import pandas as pd
from unittest.mock import patch, MagicMock
import os
from utils import inference_eval_utils

# Test for get_data_path
def test_get_data_path():
    """Test that get_data_path returns the correct path based on config.DATA_SOURCE."""
    # Create a mock config with all required paths
    mock_config = MagicMock()
    mock_config.DATA_SOURCE = 'synthetic'
    mock_config.SYNTHETIC_DATA_PATH = 'data/synthetic.csv'
    mock_config.SYNTHETIC_UPDATED_DATA_PATH = 'data/synthetic_updated.csv'
    mock_config.SAMPLE_DATA_PATH = 'data/sample.csv'
    mock_config.IMAGING_DATA_PATH = 'data/imaging.csv'
    mock_config.NOTES_DATA_PATH = 'data/notes.csv'
    mock_config.LETTERS_DATA_PATH = 'data/letters.csv'
    mock_config.NPH_DATA_PATH = 'data/nph.csv'
    
    # Test each valid data source
    assert inference_eval_utils.get_data_path(mock_config) == 'data/synthetic.csv'
    
    mock_config.DATA_SOURCE = 'synthetic_updated'
    assert inference_eval_utils.get_data_path(mock_config) == 'data/synthetic_updated.csv'
    
    mock_config.DATA_SOURCE = 'sample'
    assert inference_eval_utils.get_data_path(mock_config) == 'data/sample.csv'
    
    # Test invalid data source
    mock_config.DATA_SOURCE = 'invalid'
    with pytest.raises(ValueError):
        inference_eval_utils.get_data_path(mock_config)

# Test for safe_json_loads
def test_safe_json_loads_valid_json():
    """Test safe_json_loads with valid JSON."""
    json_str = '{"a": 1, "b": true, "c": null}'
    result = inference_eval_utils.safe_json_loads(json_str)
    assert result == {"a": 1, "b": True, "c": None}

def test_safe_json_loads_python_literals():
    """Test safe_json_loads with Python literals (single quotes, True/False/None)."""
    python_str = "{'a': 1, 'b': True, 'c': None}"
    result = inference_eval_utils.safe_json_loads(python_str)
    assert result == {"a": 1, "b": True, "c": None}

def test_safe_json_loads_invalid():
    """Test safe_json_loads with invalid JSON."""
    with pytest.raises(json.JSONDecodeError):
        inference_eval_utils.safe_json_loads("{'a': 1, 'b':")

# Test for transform_python_to_json
def test_transform_python_to_json_with_datetime():
    """Test transform_python_to_json with datetime objects."""
    python_str = "[{'date': datetime.date(2023, 4, 1), 'event': 'checkup'}]"
    expected_json = '[{"date": "2023-04-01", "event": "checkup"}]'
    result = inference_eval_utils.transform_python_to_json(python_str)
    assert json.loads(result) == json.loads(expected_json)

def test_transform_python_to_json_empty():
    """Test transform_python_to_json with empty input."""
    assert inference_eval_utils.transform_python_to_json("") == "[]"
    assert inference_eval_utils.transform_python_to_json(None) == "[]"
    assert inference_eval_utils.transform_python_to_json(pd.NA) == "[]"

# Test for preprocess_text
def test_preprocess_text():
    """Test preprocess_text function."""
    # Test lowercase conversion
    assert inference_eval_utils.preprocess_text("HELLO") == "hello"
    
    # Test special character replacement
    assert inference_eval_utils.preprocess_text("hello, world!") == "hello  world "
    
    # Test multiple spaces replacement
    assert inference_eval_utils.preprocess_text("hello    world") == "hello world"
    
    # Test that periods are preserved
    assert inference_eval_utils.preprocess_text("hello. world.") == "hello. world."

# Test for load_and_prepare_data with real test data
@patch('utils.inference_eval_utils.pd.read_csv')
def test_load_and_prepare_data_diagnosis_only(mock_read_csv, diagnosis_only_config):
    """Test load_and_prepare_data function with diagnosis_only configuration."""
    # Create a small mock dataset
    mock_df = pd.DataFrame({
        'note_id': [1, 2],
        'patient_id': ['P001', 'P002'],
        'note': [
            'Patient presents with headache for 3 months. History of pituitary adenoma diagnosed 2 years ago.',
            'Follow-up visit for visual field defects. MRI shows 3mm pituitary lesion.'
        ],
        'extracted_disorders': [
            '[{"label": "headache", "start": 23}, {"label": "pituitary adenoma", "start": 54}]',
            '[{"label": "visual field defects", "start": 20}, {"label": "pituitary lesion", "start": 49}]'
        ],
        'formatted_dates': [
            '[{"parsed": "2023-01-15", "original": "3 months", "start": 40}, {"parsed": "2021-04-15", "original": "2 years ago", "start": 80}]',
            '[]'
        ],
        'relationships_gold': [
            '[{"diagnosis": "headache", "date": "2023-01-15"}, {"diagnosis": "pituitary adenoma", "date": "2021-04-15"}]',
            '[{"diagnosis": "visual field defects", "date": "2023-04-10"}, {"diagnosis": "pituitary lesion", "date": "2023-04-10"}]'
        ]
    })
    
    # Configure the mock to return our dataframe
    mock_read_csv.return_value = mock_df
    
    # Call the function with our diagnosis_only_config fixture
    prepared_data, relationship_gold = inference_eval_utils.load_and_prepare_data(
        dataset_path=diagnosis_only_config.DATA_PATH,
        num_samples=None,
        config=diagnosis_only_config,
        data_split_mode='all'
    )
    
    # Verify the results
    assert prepared_data is not None
    assert len(prepared_data) == 2  # Two notes
    assert relationship_gold is not None
    assert len(relationship_gold) > 0
    
    # Check that the entities were extracted correctly
    assert 'entities' in prepared_data[0]
    entities = prepared_data[0]['entities']
    assert isinstance(entities, tuple)
    assert len(entities) == 2  # (diagnoses, dates)

@patch('utils.inference_eval_utils.pd.read_csv')
def test_load_and_prepare_data_multi_entity(mock_read_csv, multi_entity_config):
    """Test load_and_prepare_data function with multi_entity configuration."""
    # Create a small mock dataset
    mock_df = pd.DataFrame({
        'note_id': [1, 2],
        'patient_id': ['P001', 'P002'],
        'note': [
            'Patient presents with headache for 3 months. History of pituitary adenoma diagnosed 2 years ago.',
            'Follow-up visit for visual field defects. MRI shows 3mm pituitary lesion.'
        ],
        'extracted_snomed_entities': [
            '[{"label": "headache", "categories": ["symptom"], "start": 23}]',
            '[{"label": "visual field defects", "categories": ["symptom"], "start": 20}]'
        ],
        'extracted_umls_entities': [
            '[{"label": "pituitary adenoma", "categories": ["disorder"], "start": 54}]',
            '[{"label": "pituitary lesion", "categories": ["disorder"], "start": 49}]'
        ],
        'formatted_dates': [
            '[{"parsed": "2023-01-15", "original": "3 months", "start": 40}, {"parsed": "2021-04-15", "original": "2 years ago", "start": 80}]',
            '[]'
        ],
        'entity_gold': [
            '[{"entity_label": "headache", "entity_category": "symptom", "start": 23, "end": 31}, {"entity_label": "pituitary adenoma", "entity_category": "disorder", "start": 54, "end": 71}]',
            '[{"entity_label": "visual field defects", "entity_category": "symptom", "start": 20, "end": 39}, {"entity_label": "pituitary lesion", "entity_category": "disorder", "start": 49, "end": 65}]'
        ],
        'relationship_gold': [
            '[{"entity_label": "headache", "entity_category": "symptom", "date": "2023-01-15"}, {"entity_label": "pituitary adenoma", "entity_category": "disorder", "date": "2021-04-15"}]',
            '[{"entity_label": "visual field defects", "entity_category": "symptom", "date": "2023-04-10"}, {"entity_label": "pituitary lesion", "entity_category": "disorder", "date": "2023-04-10"}]'
        ]
    })
    
    # Configure the mock to return our dataframe
    mock_read_csv.return_value = mock_df
    
    # Call the function with our multi_entity_config fixture
    prepared_data, entity_gold, relationship_gold, pa_likelihood_gold = inference_eval_utils.load_and_prepare_data(
        dataset_path=multi_entity_config.DATA_PATH,
        num_samples=None,
        config=multi_entity_config,
        data_split_mode='all'
    )
    
    # Verify the results
    assert prepared_data is not None
    assert len(prepared_data) == 2  # Two notes
    assert entity_gold is not None
    assert len(entity_gold) > 0
    assert relationship_gold is not None
    assert len(relationship_gold) > 0
    
    # Check that the entities were extracted correctly
    assert 'entities' in prepared_data[0]
    entities = prepared_data[0]['entities']
    assert isinstance(entities, tuple)
    assert len(entities) == 2  # (entities_list, dates_list)

# Test for preprocess_note_for_prediction
def test_preprocess_note_for_prediction():
    """Test preprocess_note_for_prediction function."""
    note = "Patient presents with headache for 3 months. History of pituitary adenoma diagnosed 2 years ago."
    diagnoses = [("headache", 23), ("pituitary adenoma", 54)]
    dates = [("2023-01-15", "3 months", 40), ("2021-04-15", "2 years ago", 80)]
    
    features = inference_eval_utils.preprocess_note_for_prediction(note, diagnoses, dates, MAX_DISTANCE=500)
    
    # Should create 4 features (2 diagnoses x 2 dates)
    assert len(features) == 4
    
    # Check the first feature (headache - 3 months)
    assert features[0]['diagnosis'] == 'headache'
    assert features[0]['date'] == '2023-01-15'
    assert 'context' in features[0]
    assert features[0]['distance'] == 40 - 23  # date_pos - diag_pos
    assert features[0]['diag_before_date'] == 1  # diag_pos < date_pos 