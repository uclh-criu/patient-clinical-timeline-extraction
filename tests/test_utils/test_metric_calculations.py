import pytest
import os
import sys
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

# Add project root to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import the utility functions to test
from utils.inference_eval_utils import (
    calculate_entity_metrics,
    calculate_and_report_metrics
)

# Create a simple function for testing precision, recall, and F1 calculation
def calculate_precision_recall_f1(predictions, gold_standard):
    """
    Calculate precision, recall, and F1 score for a set of predictions against a gold standard.
    This is a simplified version of the logic in calculate_and_report_metrics.
    
    Args:
        predictions (list): List of predicted relationships.
        gold_standard (list): List of gold standard relationships.
        
    Returns:
        dict: A dictionary containing precision, recall, F1 score, and accuracy.
    """
    # Create sets for comparison
    pred_set = set()
    gold_set = set()
    
    # Convert predictions to a set of tuples for comparison
    for p in predictions:
        if 'entity_label' in p:
            pred_set.add((p['note_id'], p['entity_label'], p['date']))
        elif 'diagnosis' in p:
            pred_set.add((p['note_id'], p['diagnosis'], p['date']))
    
    # Convert gold standard to a set of tuples for comparison
    for g in gold_standard:
        if 'entity_label' in g:
            gold_set.add((g['note_id'], g['entity_label'], g['date']))
        elif 'diagnosis' in g:
            gold_set.add((g['note_id'], g['diagnosis'], g['date']))
    
    # Calculate metrics
    true_positives = len(pred_set & gold_set)
    false_positives = len(pred_set - gold_set)
    false_negatives = len(gold_set - pred_set)
    
    # Calculate precision, recall, and F1
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = true_positives / (true_positives + false_positives + false_negatives) if (true_positives + false_positives + false_negatives) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy
    }

class TestMetricCalculations:
    """Tests for metric calculation functions."""
    
    @pytest.fixture
    def sample_predictions(self):
        """Create sample predictions for testing metrics."""
        return [
            {'note_id': 1, 'entity_label': 'headache', 'entity_category': 'disorder', 'date': '2023-01-15'},
            {'note_id': 1, 'entity_label': 'fever', 'entity_category': 'disorder', 'date': '2023-01-10'},
            {'note_id': 2, 'entity_label': 'cough', 'entity_category': 'disorder', 'date': '2023-01-20'},
            {'note_id': 3, 'entity_label': 'fatigue', 'entity_category': 'disorder', 'date': '2023-02-01'},
            {'note_id': 4, 'entity_label': 'pituitary adenoma', 'entity_category': 'disorder', 'date': '2023-03-15'}
        ]
    
    @pytest.fixture
    def sample_gold_standard(self):
        """Create sample gold standard for testing metrics."""
        return [
            {'note_id': 1, 'entity_label': 'headache', 'entity_category': 'disorder', 'date': '2023-01-15'},
            {'note_id': 1, 'entity_label': 'fever', 'entity_category': 'disorder', 'date': '2023-01-12'},  # Date mismatch
            {'note_id': 2, 'entity_label': 'cough', 'entity_category': 'disorder', 'date': '2023-01-20'},
            {'note_id': 3, 'entity_label': 'dizziness', 'entity_category': 'disorder', 'date': '2023-02-05'},  # Entity mismatch
            {'note_id': 5, 'entity_label': 'migraine', 'entity_category': 'disorder', 'date': '2023-04-01'}  # Missing in predictions
        ]
    
    def test_calculate_precision_recall_f1_perfect_match(self):
        """Test metric calculation with perfect matching predictions and gold standard."""
        predictions = [
            {'note_id': 1, 'entity_label': 'headache', 'date': '2023-01-15'},
            {'note_id': 2, 'entity_label': 'fever', 'date': '2023-01-10'}
        ]
        
        gold_standard = [
            {'note_id': 1, 'entity_label': 'headache', 'date': '2023-01-15'},
            {'note_id': 2, 'entity_label': 'fever', 'date': '2023-01-10'}
        ]
        
        metrics = calculate_precision_recall_f1(predictions, gold_standard)
        
        # Perfect match should give precision = recall = f1 = 1.0
        assert metrics['precision'] == 1.0
        assert metrics['recall'] == 1.0
        assert metrics['f1'] == 1.0
        assert metrics['accuracy'] == 1.0
    
    def test_calculate_precision_recall_f1_no_match(self):
        """Test metric calculation with no matching predictions and gold standard."""
        predictions = [
            {'note_id': 1, 'entity_label': 'headache', 'date': '2023-01-15'},
            {'note_id': 2, 'entity_label': 'fever', 'date': '2023-01-10'}
        ]
        
        gold_standard = [
            {'note_id': 3, 'entity_label': 'cough', 'date': '2023-01-20'},
            {'note_id': 4, 'entity_label': 'fatigue', 'date': '2023-02-01'}
        ]
        
        metrics = calculate_precision_recall_f1(predictions, gold_standard)
        
        # No match should give precision = recall = f1 = 0.0
        assert metrics['precision'] == 0.0
        assert metrics['recall'] == 0.0
        assert metrics['f1'] == 0.0
        assert metrics['accuracy'] == 0.0
    
    def test_calculate_precision_recall_f1_partial_match(self, sample_predictions, sample_gold_standard):
        """Test metric calculation with partially matching predictions and gold standard."""
        metrics = calculate_precision_recall_f1(sample_predictions, sample_gold_standard)
        
        # Expected values:
        # True Positives = 2 (headache and cough match perfectly)
        # False Positives = 3 (fever date mismatch, fatigue entity mismatch, pituitary adenoma not in gold)
        # False Negatives = 3 (fever date mismatch, dizziness entity mismatch, migraine not in predictions)
        
        # Precision = TP / (TP + FP) = 2 / (2 + 3) = 2 / 5 = 0.4
        # Recall = TP / (TP + FN) = 2 / (2 + 3) = 2 / 5 = 0.4
        # F1 = 2 * (Precision * Recall) / (Precision + Recall) = 2 * (0.4 * 0.4) / (0.4 + 0.4) = 0.4
        
        assert metrics['precision'] == pytest.approx(0.4)
        assert metrics['recall'] == pytest.approx(0.4)
        assert metrics['f1'] == pytest.approx(0.4)
    
    def test_calculate_precision_recall_f1_empty_predictions(self, sample_gold_standard):
        """Test metric calculation with empty predictions."""
        metrics = calculate_precision_recall_f1([], sample_gold_standard)
        
        # No predictions should give precision = 0.0, recall = 0.0, f1 = 0.0
        assert metrics['precision'] == 0.0
        assert metrics['recall'] == 0.0
        assert metrics['f1'] == 0.0
    
    def test_calculate_precision_recall_f1_empty_gold_standard(self, sample_predictions):
        """Test metric calculation with empty gold standard."""
        metrics = calculate_precision_recall_f1(sample_predictions, [])
        
        # No gold standard should give precision = 0.0, recall = 0.0, f1 = 0.0
        assert metrics['precision'] == 0.0
        assert metrics['recall'] == 0.0
        assert metrics['f1'] == 0.0
    
    def test_calculate_entity_metrics(self):
        """Test entity metrics calculation."""
        # Create sample prepared data
        prepared_data = [
            {'note_id': 1, 'entities': ([('headache', 10), ('fever', 20)], []),
             'extracted_entities': [
                {'label': 'headache', 'category': 'disorder', 'start': 10, 'end': 18},
                {'label': 'fever', 'category': 'disorder', 'start': 20, 'end': 25}
             ]},
            {'note_id': 2, 'entities': ([('cough', 15)], []),
             'extracted_entities': [
                {'label': 'cough', 'category': 'disorder', 'start': 15, 'end': 20}
             ]},
            {'note_id': 3, 'entities': ([('fatigue', 25)], []),
             'extracted_entities': [
                {'label': 'fatigue', 'category': 'disorder', 'start': 25, 'end': 32}
             ]}
        ]
        
        # Create sample entity gold standard
        entity_gold = [
            {'note_id': 1, 'entity_label': 'headache', 'entity_category': 'disorder', 'start': 10, 'end': 18},
            {'note_id': 1, 'entity_label': 'fever', 'entity_category': 'disorder', 'start': 20, 'end': 25},
            {'note_id': 2, 'entity_label': 'cough', 'entity_category': 'disorder', 'start': 15, 'end': 20},
            {'note_id': 3, 'entity_label': 'dizziness', 'entity_category': 'disorder', 'start': 25, 'end': 33}  # Mismatch
        ]
        
        # Mock the output directory
        output_dir = 'test_output'
        os.makedirs(output_dir, exist_ok=True)
        
        # Call the function with mocked dependencies
        with patch('utils.inference_eval_utils.plt.figure'), \
             patch('utils.inference_eval_utils.ConfusionMatrixDisplay'), \
             patch('utils.inference_eval_utils.plt.savefig'), \
             patch('utils.inference_eval_utils.plt.close'):
            
            # Call the function
            metrics = calculate_entity_metrics(prepared_data, entity_gold, output_dir)
            
            # Check that the metrics were calculated
            assert 'precision' in metrics
            assert 'recall' in metrics
            assert 'f1' in metrics
            
            # We expect 3/4 correct entities (75% precision and recall)
            assert metrics['precision'] == pytest.approx(0.75)
            assert metrics['recall'] == pytest.approx(0.75)
            assert metrics['f1'] == pytest.approx(0.75)
    
    def test_calculate_and_report_metrics(self):
        """Test relationship metrics calculation and reporting."""
        # Create sample predictions
        predictions = [
            {'note_id': 1, 'entity_label': 'headache', 'date': '2023-01-15'},
            {'note_id': 2, 'entity_label': 'fever', 'date': '2023-01-10'}
        ]
        
        # Create sample gold standard
        gold_standard = [
            {'note_id': 1, 'entity_label': 'headache', 'date': '2023-01-15'},
            {'note_id': 2, 'entity_label': 'fever', 'date': '2023-01-12'}  # Date mismatch
        ]
        
        # Mock the output directory and other dependencies
        output_dir = 'test_output'
        os.makedirs(output_dir, exist_ok=True)
        
        # Call the function with mocked dependencies
        with patch('utils.inference_eval_utils.plt.figure'), \
             patch('utils.inference_eval_utils.ConfusionMatrixDisplay'), \
             patch('utils.inference_eval_utils.plt.savefig'), \
             patch('utils.inference_eval_utils.plt.close'), \
             patch('utils.inference_eval_utils.pd.DataFrame'), \
             patch('utils.inference_eval_utils.pd.read_csv'):
            
            # Call the function
            metrics = calculate_and_report_metrics(
                predictions, 
                gold_standard, 
                'test_extractor', 
                output_dir, 
                2,  # num_notes
                'test_dataset_path'
            )
            
            # Check that the metrics were calculated
            assert 'precision' in metrics
            assert 'recall' in metrics
            assert 'f1' in metrics
            
            # We expect 1/2 correct relationships (50% precision and recall)
            assert metrics['precision'] == pytest.approx(0.5)
            assert metrics['recall'] == pytest.approx(0.5)
            assert metrics['f1'] == pytest.approx(0.5) 