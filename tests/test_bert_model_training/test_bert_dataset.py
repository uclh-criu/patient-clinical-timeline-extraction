import pytest
import torch
import pandas as pd
import json
from unittest.mock import patch, MagicMock
import os
import sys
from bert_model_training.BertEntityPairDataset import BertEntityPairDataset
from utils.inference_eval_utils import load_and_prepare_data, safe_json_loads

class TestBertEntityPairDataset:
    @pytest.fixture
    def sample_data(self, diagnosis_only_test_file_path, diagnosis_only_config):
        """Create sample data for testing the dataset from real synthetic data."""
        try:
            # Load the first few rows from the test dataset
            df = pd.read_csv(diagnosis_only_test_file_path, nrows=5)
            if len(df) == 0:
                pytest.skip("Test data file is empty")
            
            # Process the data to create BERT dataset input
            sample_data = []
            for _, row in df.iterrows():
                text = row[diagnosis_only_config.TEXT_COLUMN]
                
                # Parse diagnoses and dates
                diagnoses = safe_json_loads(row[diagnosis_only_config.DIAGNOSES_COLUMN], [])
                dates = safe_json_loads(row[diagnosis_only_config.DATES_COLUMN], [])
                
                # Create diagnosis-date pairs for the dataset
                for diag in diagnoses:
                    for date in dates:
                        # Calculate the distance between diagnosis and date
                        distance = abs(diag['start'] - date['start'])
                        
                        # Determine if diagnosis comes before date
                        diag_before_date = 1 if diag['start'] < date['start'] else 0
                        
                        # Create a context window around the entities
                        start_pos = max(0, min(diag['start'], date['start']) - 50)
                        end_pos = min(len(text), max(diag['start'] + len(diag['label']), 
                                                     date['start'] + len(date['original'])) + 50)
                        context = text[start_pos:end_pos]
                        
                        # Adjust positions relative to the context window
                        diag_pos_rel = diag['start'] - start_pos
                        date_pos_rel = date['start'] - start_pos
                        
                        # Create a sample with a random label (since we don't know the true relationship)
                        # In a real scenario, you'd use the gold standard relationships
                        sample_data.append({
                            'diagnosis': diag['label'],
                            'date': date['parsed'],
                            'context': context,
                            'distance': distance,
                            'diag_pos_rel': diag_pos_rel,
                            'date_pos_rel': date_pos_rel,
                            'diag_before_date': diag_before_date,
                            'label': 1 if distance < 100 else 0  # Simplified heuristic for testing
                        })
            
            # Return at least 2 samples or skip if not enough data
            if len(sample_data) >= 2:
                return sample_data[:2]  # Just use the first 2 for testing
            else:
                pytest.skip("Not enough diagnosis-date pairs in test data")
                
        except Exception as e:
            pytest.skip(f"Error preparing test data: {e}")
    
    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock BERT tokenizer."""
        mock = MagicMock()
        
        # Mock the encode_plus method
        def mock_encode_plus(text, **kwargs):
            # Return a simple encoding with fixed length
            return {
                'input_ids': torch.tensor([101, 102, 103, 104, 105] + [0] * 59),  # [CLS], [SEP], etc.
                'attention_mask': torch.tensor([1, 1, 1, 1, 1] + [0] * 59),
                'token_type_ids': torch.tensor([0, 0, 0, 0, 0] + [0] * 59)
            }
        
        mock.encode_plus.side_effect = mock_encode_plus
        return mock
    
    def test_init(self, sample_data, mock_tokenizer):
        """Test dataset initialization."""
        dataset = BertEntityPairDataset(sample_data, mock_tokenizer, max_length=64)
        
        # Check that the dataset has the correct length
        assert len(dataset) == len(sample_data)
        
        # Check that the tokenizer was used
        assert dataset.tokenizer == mock_tokenizer
        
        # Check that the max_length was set correctly
        assert dataset.max_length == 64
    
    def test_getitem(self, sample_data, mock_tokenizer):
        """Test retrieving an item from the dataset."""
        dataset = BertEntityPairDataset(sample_data, mock_tokenizer, max_length=64)
        
        # Get the first item
        item = dataset[0]
        
        # Check that the item has the expected keys
        assert 'input_ids' in item
        assert 'attention_mask' in item
        assert 'token_type_ids' in item
        assert 'labels' in item
        assert 'distance' in item
        assert 'diag_before' in item
        
        # Check that the tensors have the expected shapes
        assert item['input_ids'].shape == torch.Size([64])
        assert item['attention_mask'].shape == torch.Size([64])
        assert item['token_type_ids'].shape == torch.Size([64])
        assert item['labels'].shape == torch.Size([])
        assert item['distance'].shape == torch.Size([1])
        assert item['diag_before'].shape == torch.Size([1])
        
        # Check that the label is correct
        assert item['labels'].item() == sample_data[0]['label']
        
        # Check that the distance is normalized
        assert 0 <= item['distance'].item() <= 1
        
        # Check that diag_before is correct
        assert item['diag_before'].item() == sample_data[0]['diag_before_date']
    
    def test_len(self, sample_data, mock_tokenizer):
        """Test the length of the dataset."""
        dataset = BertEntityPairDataset(sample_data, mock_tokenizer, max_length=64)
        assert len(dataset) == len(sample_data)
    
    def test_empty_dataset(self, mock_tokenizer):
        """Test initializing an empty dataset."""
        dataset = BertEntityPairDataset([], mock_tokenizer, max_length=64)
        assert len(dataset) == 0
    
    def test_prepare_features(self, sample_data, mock_tokenizer):
        """Test the _prepare_features method."""
        dataset = BertEntityPairDataset(sample_data, mock_tokenizer, max_length=64)
        
        # Call _prepare_features directly
        features = dataset._prepare_features(sample_data[0])
        
        # Check that the features have the expected keys
        assert 'input_ids' in features
        assert 'attention_mask' in features
        assert 'token_type_ids' in features
        assert 'labels' in features
        assert 'distance' in features
        assert 'diag_before' in features
        
        # Check that the tokenizer was called with the context
        mock_tokenizer.encode_plus.assert_called_with(
            sample_data[0]['context'],
            add_special_tokens=True,
            max_length=64,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=True,
            return_tensors='pt'
        )
    
    def test_normalize_distance(self, sample_data, mock_tokenizer):
        """Test distance normalization."""
        dataset = BertEntityPairDataset(sample_data, mock_tokenizer, max_length=64, max_distance=100)
        
        # Call _prepare_features and check the normalized distance
        features = dataset._prepare_features(sample_data[0])
        assert features['distance'].item() == pytest.approx(min(sample_data[0]['distance'] / 100, 1.0))
        
        # Test with distance > max_distance
        sample_with_large_distance = sample_data[0].copy()
        sample_with_large_distance['distance'] = 150
        features = dataset._prepare_features(sample_with_large_distance)
        assert features['distance'].item() == 1.0  # Should be capped at 1.0 