import pytest
import torch
import torch.nn as nn
from custom_model_training.DiagnosisDateRelationModel import DiagnosisDateRelationModel

class TestDiagnosisDateRelationModel:
    @pytest.fixture
    def model_params(self):
        """Define default model parameters for testing."""
        return {
            'vocab_size': 1000,
            'embedding_dim': 64,
            'hidden_dim': 32,
            'dropout': 0.2
        }
    
    def test_init(self, model_params):
        """Test model initialization with default parameters."""
        model = DiagnosisDateRelationModel(**model_params)
        
        # Check that the model components exist
        assert hasattr(model, 'embedding')
        assert hasattr(model, 'lstm')
        assert hasattr(model, 'fc1')
        assert hasattr(model, 'fc2')
        assert hasattr(model, 'dropout')
        assert hasattr(model, 'sigmoid')
        
        # Check component types
        assert isinstance(model.embedding, nn.Embedding)
        assert isinstance(model.lstm, nn.LSTM)
        assert isinstance(model.fc1, nn.Linear)
        assert isinstance(model.fc2, nn.Linear)
        assert isinstance(model.dropout, nn.Dropout)
        assert isinstance(model.sigmoid, nn.Sigmoid)
        
        # Check component dimensions
        assert model.embedding.num_embeddings == model_params['vocab_size']
        assert model.embedding.embedding_dim == model_params['embedding_dim']
        assert model.lstm.input_size == model_params['embedding_dim']
        assert model.lstm.hidden_size == model_params['hidden_dim']
        assert model.fc1.in_features == model_params['hidden_dim'] + 2  # hidden_dim + distance + diag_before
        assert model.fc1.out_features == model_params['hidden_dim']
        assert model.fc2.in_features == model_params['hidden_dim']
        assert model.fc2.out_features == 1
        assert model.dropout.p == model_params['dropout']
    
    def test_forward(self, model_params):
        """Test the forward pass of the model."""
        model = DiagnosisDateRelationModel(**model_params)
        
        # Create dummy input
        batch_size = 4
        seq_len = 50
        tokens = torch.randint(0, model_params['vocab_size'], (batch_size, seq_len))
        distance = torch.rand(batch_size, 1)
        diag_before = torch.randint(0, 2, (batch_size, 1)).float()
        
        # Run forward pass
        output = model(tokens, distance, diag_before)
        
        # Check output shape
        assert output.shape == torch.Size([batch_size, 1])
        
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
        distance = torch.zeros(batch_size, 1)
        diag_before = torch.zeros(batch_size, 1)
        
        # Run forward pass
        output = model(tokens, distance, diag_before)
        
        # Check output shape
        assert output.shape == torch.Size([batch_size, 1])
        
        # Check output range
        assert torch.all(output >= 0)
        assert torch.all(output <= 1)
    
    def test_model_with_different_params(self):
        """Test model initialization with different parameters."""
        # Test with different embedding dimensions
        model = DiagnosisDateRelationModel(vocab_size=500, embedding_dim=32, hidden_dim=16, dropout=0.1)
        assert model.embedding.num_embeddings == 500
        assert model.embedding.embedding_dim == 32
        assert model.lstm.hidden_size == 16
        assert model.dropout.p == 0.1
        
        # Test with no dropout
        model = DiagnosisDateRelationModel(vocab_size=500, embedding_dim=32, hidden_dim=16, dropout=0.0)
        assert model.dropout.p == 0.0
    
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
        distance = torch.rand(batch_size, 1)
        diag_before = torch.randint(0, 2, (batch_size, 1)).float()
        targets = torch.randint(0, 2, (batch_size, 1)).float()
        
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
        distance = torch.rand(batch_size, 1)
        diag_before = torch.randint(0, 2, (batch_size, 1)).float()
        
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