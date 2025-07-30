import pytest
import torch
import torch.nn as nn
import os
import sys
from custom_model_training.DiagnosisDateRelationModel import DiagnosisDateRelationModel

# Add project root to path to allow importing from other directories
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

class TestDiagnosisDateRelationModel:
    @pytest.fixture
    def model_params(self):
        """Define default model parameters for testing."""
        return {
            'vocab_size': 1000,
            'embedding_dim': 64,
            'hidden_dim': 32,
            'apply_sigmoid': True
        }
    
    def test_init(self, model_params):
        """Test model initialization with default parameters."""
        model = DiagnosisDateRelationModel(**model_params)
        
        # Check that the model components exist
        assert hasattr(model, 'embedding')
        assert hasattr(model, 'entity_category_embedding')
        assert hasattr(model, 'conv1')
        assert hasattr(model, 'conv2')
        assert hasattr(model, 'pool')
        assert hasattr(model, 'lstm')
        assert hasattr(model, 'fc1')
        assert hasattr(model, 'fc2')
        assert hasattr(model, 'dropout')
        assert hasattr(model, 'apply_sigmoid')
        
        # Check component types
        assert isinstance(model.embedding, nn.Embedding)
        assert isinstance(model.entity_category_embedding, nn.Embedding)
        assert isinstance(model.conv1, nn.Conv1d)
        assert isinstance(model.conv2, nn.Conv1d)
        assert isinstance(model.pool, nn.MaxPool1d)
        assert isinstance(model.lstm, nn.LSTM)
        assert isinstance(model.fc1, nn.Linear)
        assert isinstance(model.fc2, nn.Linear)
        assert isinstance(model.dropout, nn.Dropout)
        assert isinstance(model.apply_sigmoid, bool)
        
        # Check component dimensions
        assert model.embedding.num_embeddings == model_params['vocab_size']
        assert model.embedding.embedding_dim == model_params['embedding_dim']
        assert model.lstm.hidden_size == model_params['hidden_dim']
        # Note: The model architecture has changed since these tests were written
        # Now using CNN + LSTM with different dimensions
        assert model.dropout.p == 0.3  # Hardcoded in the model
    
    def test_forward(self, model_params):
        """Test the forward pass of the model."""
        model = DiagnosisDateRelationModel(**model_params)
        
        # Create dummy input
        batch_size = 4
        seq_len = 50
        tokens = torch.randint(0, model_params['vocab_size'], (batch_size, seq_len))
        distance = torch.rand(batch_size)
        diag_before = torch.randint(0, 2, (batch_size,)).float()
        
        # Run forward pass
        output = model(tokens, distance, diag_before)
        
        # Check output shape - model now returns a flattened tensor
        assert output.shape == torch.Size([batch_size])
        
        # Check output range (sigmoid output should be between 0 and 1)
        assert torch.all(output >= 0)
        assert torch.all(output <= 1)
    
    def test_forward_with_empty_sequence(self, model_params):
        """Test the forward pass with an empty sequence (all zeros)."""
        model = DiagnosisDateRelationModel(**model_params)
        
        # Create dummy input with all zeros
        batch_size = 2
        seq_len = 50
        tokens = torch.zeros(batch_size, seq_len).long()
        distance = torch.zeros(batch_size)
        diag_before = torch.zeros(batch_size)
        
        # Run forward pass
        output = model(tokens, distance, diag_before)
        
        # Check output shape - model now returns a flattened tensor
        assert output.shape == torch.Size([batch_size])
        
        # Check output range
        assert torch.all(output >= 0)
        assert torch.all(output <= 1)
    
    def test_model_with_different_params(self):
        """Test model initialization with different parameters."""
        # Test with different embedding dimensions
        model = DiagnosisDateRelationModel(vocab_size=500, embedding_dim=32, hidden_dim=16)
        assert model.embedding.num_embeddings == 500
        assert model.embedding.embedding_dim == 32
        assert model.lstm.hidden_size == 16
        assert model.dropout.p == 0.3  # Hardcoded in the model
        
        # Test with apply_sigmoid=False
        model = DiagnosisDateRelationModel(vocab_size=500, embedding_dim=32, hidden_dim=16, apply_sigmoid=False)
        assert model.apply_sigmoid == False
    
    def test_model_trainable(self, model_params):
        """Test that the model is trainable with gradient descent."""
        model = DiagnosisDateRelationModel(**model_params)
        
        # Set up a simple training loop
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.BCELoss()
        
        # Create dummy data
        batch_size = 4
        seq_len = 50
        tokens = torch.randint(0, model_params['vocab_size'], (batch_size, seq_len))
        distance = torch.rand(batch_size)
        diag_before = torch.randint(0, 2, (batch_size,)).float()
        targets = torch.randint(0, 2, (batch_size,)).float()
        
        # Initial loss
        model.train()
        optimizer.zero_grad()
        outputs = model(tokens, distance, diag_before)
        initial_loss = criterion(outputs, targets)
        
        # Backpropagation and optimization
        initial_loss.backward()
        optimizer.step()
        
        # Second forward pass
        optimizer.zero_grad()
        outputs = model(tokens, distance, diag_before)
        second_loss = criterion(outputs, targets)
        
        # The loss should decrease after optimization
        assert second_loss < initial_loss
    
    def test_model_save_load(self, model_params, tmp_path):
        """Test saving and loading the model."""
        model = DiagnosisDateRelationModel(**model_params)
        
        # Create a dummy input
        batch_size = 2
        seq_len = 50
        tokens = torch.randint(0, model_params['vocab_size'], (batch_size, seq_len))
        distance = torch.rand(batch_size)  # Remove the extra dimension
        diag_before = torch.randint(0, 2, (batch_size,)).float()  # Remove the extra dimension
        
        # Get output before saving
        model.eval()
        with torch.no_grad():
            output_before = model(tokens, distance, diag_before)
        
        # Save the model
        save_path = tmp_path / "test_model.pt"
        torch.save(model.state_dict(), save_path)
        
        # Create a new model with the same parameters
        new_model = DiagnosisDateRelationModel(**model_params)
        
        # Load the saved state
        new_model.load_state_dict(torch.load(save_path))
        new_model.eval()
        
        # Get output after loading
        with torch.no_grad():
            output_after = new_model(tokens, distance, diag_before)
        
        # Outputs should be identical
        assert torch.allclose(output_before, output_after)
        
    def test_save_load_with_hyperparams_and_threshold(self, model_params, tmp_path):
        """Test saving and loading the model with hyperparameters and threshold."""
        model = DiagnosisDateRelationModel(**model_params)
        
        # Create a dummy input
        batch_size = 2
        seq_len = 50
        tokens = torch.randint(0, model_params['vocab_size'], (batch_size, seq_len))
        distance = torch.rand(batch_size)
        diag_before = torch.randint(0, 2, (batch_size,)).float()
        
        # Get output before saving
        model.eval()
        with torch.no_grad():
            output_before = model(tokens, distance, diag_before)
        
        # Create test hyperparameters
        test_hyperparams = {
            'ENTITY_MODE': 'multi_entity',
            'MAX_DISTANCE': 50,
            'MAX_CONTEXT_LEN': 100,
            'EMBEDDING_DIM': model_params['embedding_dim'],
            'HIDDEN_DIM': model_params['hidden_dim'],
            'BATCH_SIZE': 32,
            'LEARNING_RATE': 0.001,
            'NUM_EPOCHS': 10,
            'DROPOUT': 0.3,  # Hardcoded in the model
            'USE_DISTANCE_FEATURE': True,
            'USE_POSITION_FEATURE': True,
            'ENTITY_CATEGORY_EMBEDDING_DIM': 8,
            'USE_WEIGHTED_LOSS': False,
            'POS_WEIGHT': None,
            'APPLY_SIGMOID': model_params['apply_sigmoid']
        }
        
        # Create a dictionary to save both model state and hyperparameters
        test_threshold = 0.75
        save_dict = {
            'hyperparameters': test_hyperparams,
            'model_state_dict': model.state_dict(),
            'vocab_size': model_params['vocab_size'],
            'best_threshold': test_threshold
        }
        
        # Save the model with hyperparameters
        save_path = tmp_path / "test_model_with_hyperparams.pt"
        torch.save(save_dict, save_path)
        
        # Create a new model with default parameters (will be overridden by loaded hyperparameters)
        new_model = DiagnosisDateRelationModel(
            vocab_size=100,  # Intentionally different from the saved model
            embedding_dim=32,
            hidden_dim=16,
            apply_sigmoid=True
        )
        
        # Load the saved state and hyperparameters
        loaded_data = torch.load(save_path)
        
        # Check that the hyperparameters were saved correctly
        assert loaded_data['hyperparameters']['EMBEDDING_DIM'] == model_params['embedding_dim']
        assert loaded_data['hyperparameters']['HIDDEN_DIM'] == model_params['hidden_dim']
        assert loaded_data['hyperparameters']['DROPOUT'] == 0.3  # Hardcoded in the model
        assert loaded_data['hyperparameters']['APPLY_SIGMOID'] == model_params['apply_sigmoid']
        assert loaded_data['vocab_size'] == model_params['vocab_size']
        assert loaded_data['best_threshold'] == test_threshold
        
        # Create a new model with the loaded hyperparameters
        new_model = DiagnosisDateRelationModel(
            vocab_size=loaded_data['vocab_size'],
            embedding_dim=loaded_data['hyperparameters']['EMBEDDING_DIM'],
            hidden_dim=loaded_data['hyperparameters']['HIDDEN_DIM'],
            apply_sigmoid=loaded_data['hyperparameters']['APPLY_SIGMOID']
        )
        
        # Load the saved state
        new_model.load_state_dict(loaded_data['model_state_dict'])
        new_model.eval()
        
        # Get output after loading
        with torch.no_grad():
            output_after = new_model(tokens, distance, diag_before)
        
        # Outputs should be identical
        assert torch.allclose(output_before, output_after)
        
        # Test threshold-based prediction
        with torch.no_grad():
            raw_predictions = new_model(tokens, distance, diag_before)
            thresholded_predictions = (raw_predictions >= test_threshold).float()
            
        # Check that thresholding works as expected
        for i in range(len(raw_predictions)):
            if raw_predictions[i] >= test_threshold:
                assert thresholded_predictions[i] == 1.0
            else:
                assert thresholded_predictions[i] == 0.0 