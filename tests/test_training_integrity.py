import pytest
import os
import pandas as pd
import numpy as np
import torch
import random
from unittest.mock import patch, MagicMock
import sys

# Add project root to path for imports
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import utility functions and models
from utils.training_utils import generate_hyperparameter_grid
from utils.inference_eval_utils import load_and_prepare_data
from custom_model_training.DiagnosisDateRelationModel import DiagnosisDateRelationModel
from custom_model_training.ClinicalNoteDataset import ClinicalNoteDataset
from custom_model_training.Vocabulary import Vocabulary

class TestDataSplitting:
    """Tests to ensure data splitting is done correctly and reproducibly."""
    
    def test_patient_split_ratio(self, diagnosis_only_test_file_path, diagnosis_only_config):
        """Test that patients are split according to the specified ratio."""
        # Load data
        df = pd.read_csv(diagnosis_only_test_file_path)
        
        # Get total number of unique patients
        patient_id_column = diagnosis_only_config.PATIENT_ID_COLUMN
        total_unique_patients = len(df[patient_id_column].unique())
        
        # Set the split ratio in the config
        config = diagnosis_only_config
        
        # Load and prepare data with 'train' mode
        train_data, _ = load_and_prepare_data(
            config.DATA_PATH, 
            None,  # Use all samples
            config, 
            data_split_mode='train'
        )
        
        # Load and prepare data with 'test' mode
        test_data, _ = load_and_prepare_data(
            config.DATA_PATH, 
            None,  # Use all samples
            config, 
            data_split_mode='test'
        )
        
        # Extract unique patient IDs from train and test data
        train_patient_ids = set(item.get('patient_id') for item in train_data if item.get('patient_id') is not None)
        test_patient_ids = set(item.get('patient_id') for item in test_data if item.get('patient_id') is not None)
        
        # Check that the patient split ratio is correct (within a small margin of error)
        expected_train_patients = int(total_unique_patients * config.TRAINING_SET_RATIO)
        assert abs(len(train_patient_ids) - expected_train_patients) <= 1
        assert abs(len(test_patient_ids) - (total_unique_patients - expected_train_patients)) <= 1
        
        # Check that all patients are accounted for
        assert len(train_patient_ids) + len(test_patient_ids) == total_unique_patients
        
        # The total number of notes might not exactly match the split ratio anymore
        # since we're splitting by patient, but we still check that all notes are accounted for
        assert len(train_data) + len(test_data) == len(df)
    
    def test_no_patient_leakage(self, diagnosis_only_test_file_path, diagnosis_only_config):
        """Test that there is no patient overlap between training and test sets."""
        # Load and prepare data with 'train' mode
        train_data, _ = load_and_prepare_data(
            diagnosis_only_test_file_path, 
            None,  # Use all samples
            diagnosis_only_config, 
            data_split_mode='train'
        )
        
        # Load and prepare data with 'test' mode
        test_data, _ = load_and_prepare_data(
            diagnosis_only_test_file_path, 
            None,  # Use all samples
            diagnosis_only_config, 
            data_split_mode='test'
        )
        
        # Extract note_ids from train and test data (still useful to check)
        train_note_ids = set(item['note_id'] for item in train_data)
        test_note_ids = set(item['note_id'] for item in test_data)
        
        # Check that there is no overlap between train and test note sets
        assert len(train_note_ids.intersection(test_note_ids)) == 0
        
        # Extract patient_ids from train and test data
        train_patient_ids = set(item.get('patient_id') for item in train_data if item.get('patient_id') is not None)
        test_patient_ids = set(item.get('patient_id') for item in test_data if item.get('patient_id') is not None)
        
        # Check that there is no overlap between train and test patient sets
        # This is the critical test for preventing data leakage
        assert len(train_patient_ids.intersection(test_patient_ids)) == 0
    
    def test_reproducible_split(self, diagnosis_only_test_file_path, diagnosis_only_config):
        """Test that using the same random seed produces the same data split."""
        # Run the split twice with the same seed
        train_data1, _ = load_and_prepare_data(
            diagnosis_only_test_file_path, 
            None,  # Use all samples
            diagnosis_only_config, 
            data_split_mode='train'
        )
        
        train_data2, _ = load_and_prepare_data(
            diagnosis_only_test_file_path, 
            None,  # Use all samples
            diagnosis_only_config, 
            data_split_mode='train'
        )
        
        # Extract note_ids from both runs
        train_note_ids1 = set(item['note_id'] for item in train_data1)
        train_note_ids2 = set(item['note_id'] for item in train_data2)
        
        # Check that the note splits are identical
        assert train_note_ids1 == train_note_ids2
        
        # Extract patient_ids from both runs
        train_patient_ids1 = set(item.get('patient_id') for item in train_data1 if item.get('patient_id') is not None)
        train_patient_ids2 = set(item.get('patient_id') for item in train_data2 if item.get('patient_id') is not None)
        
        # Check that the patient splits are identical
        assert train_patient_ids1 == train_patient_ids2
    
    def test_stratification(self, diagnosis_only_test_file_path, diagnosis_only_config):
        """Test that the data split maintains similar distribution of positive relationships."""
        import pytest
        
        # Load the full dataset to analyze the distribution of positive relationships
        df = pd.read_csv(diagnosis_only_test_file_path)
        
        # Get the patient ID column name
        patient_id_column = diagnosis_only_config.PATIENT_ID_COLUMN
        relationship_gold_column = diagnosis_only_config.RELATIONSHIP_GOLD_COLUMN
        
        # Skip test if the necessary columns are not available
        if relationship_gold_column not in df.columns:
            pytest.skip(f"Relationship gold column '{relationship_gold_column}' not found in test data")
        
        # Identify patients with positive relationships
        # First, parse the relationship gold column to extract positive relationships
        # This assumes the column format is similar to what's handled in load_and_prepare_data
        patients_with_positive = set()
        
        # Parse the relationship gold data to find patients with positive relationships
        # This is a simplified version of what might be in load_and_prepare_data
        for _, row in df.iterrows():
            patient_id = row.get(patient_id_column)
            if not patient_id:
                continue
                
            rel_gold = row.get(relationship_gold_column)
            if not rel_gold or pd.isna(rel_gold):
                continue
                
            # If there's any relationship data for this patient, consider them positive
            # This is a simplification; in reality, you'd parse the specific relationship format
            patients_with_positive.add(patient_id)
        
        # Calculate the overall percentage of patients with positive relationships
        total_patients = len(df[patient_id_column].unique())
        overall_positive_percentage = len(patients_with_positive) / total_patients if total_patients > 0 else 0
        
        # Load the train and test splits
        train_data, _ = load_and_prepare_data(
            diagnosis_only_test_file_path, 
            None,
            diagnosis_only_config, 
            data_split_mode='train'
        )
        
        test_data, _ = load_and_prepare_data(
            diagnosis_only_test_file_path, 
            None,
            diagnosis_only_config, 
            data_split_mode='test'
        )
        
        # Extract unique patient IDs from train and test data
        train_patient_ids = set(item.get('patient_id') for item in train_data if item.get('patient_id') is not None)
        test_patient_ids = set(item.get('patient_id') for item in test_data if item.get('patient_id') is not None)
        
        # Calculate the percentage of patients with positive relationships in each split
        train_positive = len(train_patient_ids.intersection(patients_with_positive))
        train_positive_percentage = train_positive / len(train_patient_ids) if train_patient_ids else 0
        
        test_positive = len(test_patient_ids.intersection(patients_with_positive))
        test_positive_percentage = test_positive / len(test_patient_ids) if test_patient_ids else 0
        
        # Check that the percentages are approximately equal (within a reasonable margin)
        # The margin is larger for smaller datasets
        margin = 0.15  # 15% margin for small test datasets
        assert pytest.approx(train_positive_percentage, abs=margin) == overall_positive_percentage
        assert pytest.approx(test_positive_percentage, abs=margin) == overall_positive_percentage

class TestModelTraining:
    """Tests to ensure model training is correct and reproducible."""
    
    @pytest.fixture
    def tiny_dataset(self):
        """Create a tiny dataset for overfitting tests."""
        # Create a small vocabulary
        vocab = Vocabulary()
        vocab.add_word('<PAD>')
        vocab.add_word('<UNK>')
        vocab.add_word('headache')
        vocab.add_word('fever')
        vocab.add_word('patient')
        vocab.add_word('has')
        vocab.add_word('since')
        vocab.add_word('yesterday')
        
        # Create a small dataset with just a few samples
        texts = [
            "patient has headache since yesterday",
            "patient has fever since yesterday"
        ]
        
        # Tokenize the texts
        tokenized_texts = []
        for text in texts:
            tokens = text.split()
            token_ids = [vocab.word2idx.get(token, vocab.word2idx['<unk>']) for token in tokens]
            tokenized_texts.append(token_ids)
        
        # Create labels (1 for headache, 0 for fever)
        labels = [1, 0]
        
        # Create dataset samples
        samples = []
        for i, (tokens, label) in enumerate(zip(tokenized_texts, labels)):
            samples.append({
                'tokens': tokens,
                'label': label,
                'diagnosis_pos': 2,  # Position of headache/fever
                'date_pos': 4,       # Position of yesterday
                'distance': 2,       # Distance between diagnosis and date
                'diag_before_date': 1  # Diagnosis comes before date
            })
        
        return vocab, samples
    
    def test_model_can_overfit(self, tiny_dataset):
        """Test that the model can overfit on a tiny dataset."""
        vocab, samples = tiny_dataset
        
        # Create a model with small dimensions
        model = DiagnosisDateRelationModel(
            vocab_size=len(vocab.word2idx),
            embedding_dim=16,
            hidden_dim=16
        )
        
        # Set the model to training mode
        model.train()
        
        # Create optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        # Create loss function
        loss_fn = torch.nn.BCELoss()
        
        # Train the model for a few epochs
        for epoch in range(100):
            epoch_loss = 0
            
            for sample in samples:
                # Zero the gradients
                optimizer.zero_grad()
                
                # Create input tensors
                tokens = torch.tensor(sample['tokens'], dtype=torch.long).unsqueeze(0)
                # Create 1D tensors for distance and diag_before
                distance = torch.tensor([sample['distance'] / 10.0], dtype=torch.float)
                diag_before = torch.tensor([sample['diag_before_date']], dtype=torch.float)
                label = torch.tensor([sample['label']], dtype=torch.float)
                
                # Forward pass
                output = model(tokens, distance, diag_before)
                
                # Calculate loss
                loss = loss_fn(output, label)
                
                # Backward pass
                loss.backward()
                
                # Update weights
                optimizer.step()
                
                epoch_loss += loss.item()
            
            # Check if the model has overfit
            if epoch_loss < 0.1:
                break
        
        # Evaluate the model
        model.eval()
        correct = 0
        with torch.no_grad():
            for sample in samples:
                # Create input tensors
                tokens = torch.tensor(sample['tokens'], dtype=torch.long).unsqueeze(0)
                # Create 1D tensors for distance and diag_before
                distance = torch.tensor([sample['distance'] / 10.0], dtype=torch.float)
                diag_before = torch.tensor([sample['diag_before_date']], dtype=torch.float)
                
                # Forward pass
                output = model(tokens, distance, diag_before)
                
                # Check if prediction is correct
                prediction = (output.item() > 0.5)
                if prediction == sample['label']:
                    correct += 1
        
        # The model should be able to perfectly fit this tiny dataset
        assert correct == len(samples)
    
    def test_training_reproducibility(self, tiny_dataset):
        """Test that training is reproducible with the same random seed."""
        vocab, samples = tiny_dataset
        
        # Function to train a model with a fixed seed
        def train_model_with_seed(seed):
            # Set random seeds
            torch.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed)
            
            # Create a model with small dimensions
            model = DiagnosisDateRelationModel(
                vocab_size=len(vocab.word2idx),
                embedding_dim=16,
                hidden_dim=16
            )
            
            # Set the model to training mode
            model.train()
            
            # Create optimizer
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            
            # Create loss function
            loss_fn = torch.nn.BCELoss()
            
            # Train the model for a fixed number of epochs
            for epoch in range(2):  # Reduced from 10 to 2 for faster testing
                for sample in samples:
                    # Zero the gradients
                    optimizer.zero_grad()
                    
                    # Create input tensors
                    tokens = torch.tensor(sample['tokens'], dtype=torch.long).unsqueeze(0)
                    # Create 1D tensors for distance and diag_before
                    distance = torch.tensor([sample['distance'] / 10.0], dtype=torch.float)
                    diag_before = torch.tensor([sample['diag_before_date']], dtype=torch.float)
                    label = torch.tensor([sample['label']], dtype=torch.float)
                    
                    # Forward pass
                    output = model(tokens, distance, diag_before)
                    
                    # Calculate loss
                    loss = loss_fn(output, label)
                    
                    # Backward pass
                    loss.backward()
                    
                    # Update weights
                    optimizer.step()
            
            # Return the trained model's parameters
            return {name: param.clone().detach() for name, param in model.named_parameters()}
        
        # Train two models with the same seed
        seed = 42
        params1 = train_model_with_seed(seed)
        params2 = train_model_with_seed(seed)
        
        # Check that all parameters are identical
        for name in params1:
            assert torch.allclose(params1[name], params2[name])
        
        # Train a model with a different seed
        params3 = train_model_with_seed(seed + 1)
        
        # Check that at least one parameter is different
        different_params = False
        for name in params1:
            if not torch.allclose(params1[name], params3[name]):
                different_params = True
                break
        
        assert different_params 