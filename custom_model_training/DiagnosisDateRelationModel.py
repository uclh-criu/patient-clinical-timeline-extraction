import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# Add project root to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from custom_model_training.training_config_custom import ENTITY_CATEGORY_EMBEDDING_DIM

# Model architecture
class DiagnosisDateRelationModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(DiagnosisDateRelationModel, self).__init__()
        
        # Word embeddings
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Entity category embeddings (for multi-entity mode)
        # Hardcoded to 4 categories: diagnosis, symptom, procedure, medication
        self.entity_category_embedding = nn.Embedding(4, ENTITY_CATEGORY_EMBEDDING_DIM)
        
        # 1D CNN for text features
        self.conv1 = nn.Conv1d(embedding_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        
        # Max pooling
        self.pool = nn.MaxPool1d(kernel_size=2)
        
        # LSTM for sequence modeling
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)
        
        # Fully connected layers
        # +2 for distance and ordering features, +ENTITY_CATEGORY_EMBEDDING_DIM for entity category
        self.fc1 = nn.Linear(hidden_dim*2 + 2 + ENTITY_CATEGORY_EMBEDDING_DIM, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.3)
        
        # Use debug mode from config
        self.debug = config.MODEL_DEBUG_MODE
    
    def forward(self, context, distance, diag_before, entity_category=None):
        batch_size = context.size(0)
        
        # Handle the case where entity_category is not provided (backward compatibility)
        if entity_category is None:
            # Default to 0 (diagnosis) if not provided
            entity_category = torch.zeros(batch_size, dtype=torch.long, device=context.device)
        
        # Comment out all layer-by-layer diagnostics to reduce verbosity
        """
        if self.debug:
            print("\n----- MODEL LAYER-BY-LAYER DIAGNOSTICS -----")
            print(f"Batch size: {batch_size}")
            print(f"Input context shape: {context.shape}")
            # Print only first few items for readability
            if batch_size > 1:
                print(f"Input distance (first 5 items): {distance[:5]}")
                print(f"Input diag_before (first 5 items): {diag_before[:5]}")
            else:
                print(f"Input distance: {distance}")
                print(f"Input diag_before: {diag_before}")
        """
        
        # Embedding layer
        embedded = self.embedding(context)  # [batch_size, seq_len, embedding_dim]
        
        """
        if self.debug:
            print(f"After embedding shape: {embedded.shape}")
            print(f"Sample embedding values (first item): {embedded[0, 0, :5]}...")  # First 5 values of first token
        """
        
        # CNN layers (need to transpose for CNN)
        embedded = embedded.permute(0, 2, 1)  # [batch_size, embedding_dim, seq_len]
        
        """
        if self.debug:
            print(f"After permute for CNN shape: {embedded.shape}")
        """
        
        conv_output = F.relu(self.conv1(embedded))
        
        """
        if self.debug:
            print(f"After conv1 shape: {conv_output.shape}")
            print(f"Sample conv1 values (first item): {conv_output[0, 0, :5]}...")  # First 5 values of first channel
        """
        
        conv_output = self.pool(conv_output)
        
        """
        if self.debug:
            print(f"After pool1 shape: {conv_output.shape}")
        """
        
        conv_output = F.relu(self.conv2(conv_output))
        
        """
        if self.debug:
            print(f"After conv2 shape: {conv_output.shape}")
        """
        
        conv_output = self.pool(conv_output)
        
        """
        if self.debug:
            print(f"After pool2 shape: {conv_output.shape}")
            print(f"CNN output summary: {conv_output.shape} - This is the compressed representation of local text patterns")
        """
        
        # Convert back for LSTM
        conv_output = conv_output.permute(0, 2, 1)  # [batch_size, seq_len/4, hidden_dim]
        
        """
        if self.debug:
            print(f"After permute for LSTM shape: {conv_output.shape}")
        """
        
        # LSTM layer
        lstm_output, (hidden, cell) = self.lstm(conv_output)
        
        """
        if self.debug:
            print(f"LSTM output shape: {lstm_output.shape}")
            print(f"LSTM hidden state shape: {hidden.shape}")  # [num_directions, batch_size, hidden_dim]
        """
        
        # Concatenate forward and backward hidden states
        hidden = torch.cat((hidden[0,:,:], hidden[1,:,:]), dim=1)  # [batch_size, hidden_dim*2]
        
        """
        if self.debug:
            print(f"After concatenating bidirectional hidden states: {hidden.shape}")
            print(f"Sample hidden values (first item): {hidden[0, :5]}...")  # First 5 values
        """
        
        # Get entity category embeddings
        entity_embeddings = self.entity_category_embedding(entity_category)  # [batch_size, entity_embedding_dim]
        
        # Concatenate with distance, ordering features, and entity category
        distance = distance.unsqueeze(1)  # [batch_size, 1]
        diag_before = diag_before.unsqueeze(1)  # [batch_size, 1]
        combined = torch.cat((hidden, distance, diag_before, entity_embeddings), dim=1)  # [batch_size, hidden_dim*2 + 2 + entity_embedding_dim]
        
        """
        if self.debug:
            print(f"After adding distance, diag_before, and entity features: {combined.shape}")
            print(f"Combined feature vector (text + handcrafted features):")
            print(f"  - Text representation: {hidden.shape}")
            # Print only the first item's values for batch processing
            print(f"  - Distance feature (first item): {distance[0].item()}")
            print(f"  - Diag_before feature (first item): {diag_before[0].item()}")
            print(f"  - Entity category feature (first item): {entity_category[0].item()}")
        """
        
        # Fully connected layers
        output = F.relu(self.fc1(combined))
        
        """
        if self.debug:
            print(f"After first fully connected layer: {output.shape}")
        """
        
        output = self.dropout(output)
        output = self.fc2(output)
        
        # Sigmoid for binary classification and squeeze dimension 1 to match labels shape
        final_output = torch.sigmoid(output).squeeze(1)
        
        """
        if self.debug:
            print(f"Final output shape: {final_output.shape}")
            # Print only the first item's prediction for batch processing
            print(f"Final prediction value (first item): {final_output[0].item():.6f}")
            if batch_size > 1:
                print(f"Prediction range: {final_output.min().item():.6f} to {final_output.max().item():.6f}")
            print("----- END OF MODEL DIAGNOSTICS -----\n")
        """
        
        return final_output