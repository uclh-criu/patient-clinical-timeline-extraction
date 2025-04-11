import torch
import torch.nn as nn
import torch.nn.functional as F

# Model architecture
class DiagnosisDateRelationModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(DiagnosisDateRelationModel, self).__init__()
        
        # Word embeddings
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # 1D CNN for text features
        self.conv1 = nn.Conv1d(embedding_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        
        # Max pooling
        self.pool = nn.MaxPool1d(kernel_size=2)
        
        # LSTM for sequence modeling
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_dim*2 + 2, hidden_dim)  # +2 for distance and ordering features
        self.fc2 = nn.Linear(hidden_dim, 1)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, context, distance, diag_before):
        # Embedding layer
        embedded = self.embedding(context)  # [batch_size, seq_len, embedding_dim]
        
        # CNN layers (need to transpose for CNN)
        embedded = embedded.permute(0, 2, 1)  # [batch_size, embedding_dim, seq_len]
        conv_output = F.relu(self.conv1(embedded))
        conv_output = self.pool(conv_output)
        conv_output = F.relu(self.conv2(conv_output))
        conv_output = self.pool(conv_output)
        
        # Convert back for LSTM
        conv_output = conv_output.permute(0, 2, 1)  # [batch_size, seq_len/4, hidden_dim]
        
        # LSTM layer
        lstm_output, (hidden, cell) = self.lstm(conv_output)
        
        # Concatenate forward and backward hidden states
        hidden = torch.cat((hidden[0,:,:], hidden[1,:,:]), dim=1)  # [batch_size, hidden_dim*2]
        
        # Concatenate with distance and ordering features
        distance = distance.unsqueeze(1)  # [batch_size, 1]
        diag_before = diag_before.unsqueeze(1)  # [batch_size, 1]
        combined = torch.cat((hidden, distance, diag_before), dim=1)  # [batch_size, hidden_dim*2 + 2]
        
        # Fully connected layers
        output = F.relu(self.fc1(combined))
        output = self.dropout(output)
        output = self.fc2(output)
        
        # Sigmoid for binary classification and squeeze dimension 1 to match labels shape
        return torch.sigmoid(output).squeeze(1)