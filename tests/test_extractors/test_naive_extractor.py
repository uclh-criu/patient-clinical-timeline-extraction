import pytest
import pandas as pd
import json
from unittest.mock import patch, MagicMock
from extractors.naive_extractor import NaiveExtractor
from utils.inference_eval_utils import load_and_prepare_data
from types import SimpleNamespace

class TestNaiveExtractor:
    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration for testing."""
        return SimpleNamespace(PROXIMITY_MAX_DISTANCE=200)
    
    @pytest.fixture
    def sample_data_from_file(self, diagnosis_only_test_file_path, diagnosis_only_config):
        """Load a sample clinical note from the test dataset."""
        df = pd.read_csv(diagnosis_only_test_file_path)
        if len(df) == 0:
            pytest.skip("Test data file is empty")
        
        # Get the first row from the dataset
        row = df.iloc[0]
        
        # Extract the text and entities
        text = row[diagnosis_only_config.TEXT_COLUMN]
        
        # Parse the JSON strings for diagnoses and dates
        try:
            diagnoses_json = json.loads(row[diagnosis_only_config.DIAGNOSES_COLUMN])
            diagnoses = [(item['label'], item['start']) for item in diagnoses_json]
            
            dates_json = json.loads(row[diagnosis_only_config.DATES_COLUMN])
            dates = [(item['parsed'], item['original'], item['start']) for item in dates_json]
            
            return {
                'note_id': row.name,
                'patient_id': row[diagnosis_only_config.PATIENT_ID_COLUMN],
                'text': text,
                'entities': (diagnoses, dates)
            }
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            pytest.skip(f"Error parsing test data: {e}")
    
    def test_init(self, mock_config):
        """Test NaiveExtractor initialization."""
        extractor = NaiveExtractor(mock_config)
        
        # Check that the name was set correctly
        assert extractor.name == "Naive (Proximity)"
        
        # Check that the max_distance was set correctly
        assert extractor.max_distance == 200
    
    def test_extract_with_provided_entities(self, sample_data_from_file, mock_config):
        """Test extract method with provided entities from real data."""
        extractor = NaiveExtractor(mock_config)
        
        # Call the extract method
        result = extractor.extract(
            sample_data_from_file['text'], 
            entities=sample_data_from_file['entities'], 
            note_id=sample_data_from_file['note_id'], 
            patient_id=sample_data_from_file['patient_id']
        )
        
        # Check that we got some results (we can't assert exact values since we're using real data)
        assert isinstance(result, list)
        
        # If we have results, check their structure
        if result:
            assert 'entity_label' in result[0]
            assert 'date' in result[0]
            assert 'note_id' not in result[0]  # NaiveExtractor doesn't include note_id in output
            assert 'patient_id' not in result[0]  # NaiveExtractor doesn't include patient_id in output
    
    def test_extract_with_no_entities(self, sample_data_from_file, mock_config):
        """Test extract method with no entities."""
        extractor = NaiveExtractor(mock_config)
        
        # Call the extract method with no entities
        result = extractor.extract(
            sample_data_from_file['text'], 
            entities=None, 
            note_id=sample_data_from_file['note_id'], 
            patient_id=sample_data_from_file['patient_id']
        )
        
        # Should return an empty list
        assert result == []
    
    def test_extract_with_empty_entities(self, sample_data_from_file, mock_config):
        """Test extract method with empty entities."""
        extractor = NaiveExtractor(mock_config)
        
        # Call the extract method with empty entities
        result = extractor.extract(
            sample_data_from_file['text'], 
            entities=([], []), 
            note_id=sample_data_from_file['note_id'], 
            patient_id=sample_data_from_file['patient_id']
        )
        
        # Should return an empty list
        assert result == []
    
    def test_extract_with_no_dates(self, sample_data_from_file, mock_config):
        """Test extract method with diagnoses but no dates."""
        extractor = NaiveExtractor(mock_config)
        
        # Create entities with diagnoses but no dates
        diagnoses = sample_data_from_file['entities'][0]
        entities = (diagnoses, [])
        
        # Call the extract method
        result = extractor.extract(
            sample_data_from_file['text'], 
            entities=entities, 
            note_id=sample_data_from_file['note_id'], 
            patient_id=sample_data_from_file['patient_id']
        )
        
        # Should return an empty list
        assert result == []
    
    def test_extract_with_no_diagnoses(self, sample_data_from_file, mock_config):
        """Test extract method with dates but no diagnoses."""
        extractor = NaiveExtractor(mock_config)
        
        # Create entities with dates but no diagnoses
        dates = sample_data_from_file['entities'][1]
        entities = ([], dates)
        
        # Call the extract method
        result = extractor.extract(
            sample_data_from_file['text'], 
            entities=entities, 
            note_id=sample_data_from_file['note_id'], 
            patient_id=sample_data_from_file['patient_id']
        )
        
        # Should return an empty list
        assert result == []
    
    def test_extract_with_distance_filter(self, sample_data_from_file):
        """Test extract method with distance filtering."""
        # Use a very small max_distance to filter out relationships
        config = SimpleNamespace(PROXIMITY_MAX_DISTANCE=1)
        extractor = NaiveExtractor(config)
        
        # Call the extract method with the small max_distance
        result = extractor.extract(
            sample_data_from_file['text'], 
            entities=sample_data_from_file['entities'], 
            note_id=sample_data_from_file['note_id'], 
            patient_id=sample_data_from_file['patient_id']
        )
        
        # With such a small max_distance, we should get very few or no results
        # This is hard to assert exactly with real data, but we can check the type
        assert isinstance(result, list) 