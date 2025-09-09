import torch
import torch.nn as nn
from transformers import AutoModel

class BertRC(nn.Module):
    """
    Custom BERT model with explicit span pooling for relation extraction.
    
    This model is specifically designed for relation extraction tasks where we need to
    identify relationships between two entities (disorder and date) in clinical text.
    It uses span pooling to explicitly focus on the entity representations rather than
    relying on attention mechanisms alone.
    """
    
    def __init__(self, model_name: str, tokenizer, num_labels: int, class_weights: torch.Tensor = None, dropout_rate: float = None):
        """
        Initialize the BertRC model.
        
        Args:
            model_name: HuggingFace model name (e.g., 'emilyalsentzer/Bio_ClinicalBERT')
            tokenizer: Tokenizer with special tokens [E1], [/E1], [E2], [/E2] added
            num_labels: Number of output classes (2 for binary classification)
            class_weights: Optional class weights for imbalanced data
            dropout_rate: Optional custom dropout rate
        """
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        self.backbone.resize_token_embeddings(len(tokenizer))

        self.hidden_size = self.backbone.config.hidden_size
        # Allow custom dropout rate or use model default
        dropout_prob = dropout_rate if dropout_rate is not None else self.backbone.config.hidden_dropout_prob
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(2 * self.hidden_size, num_labels)

        # Cache token IDs for markers
        self.e1_open_id = tokenizer.convert_tokens_to_ids("[E1]")
        self.e1_close_id = tokenizer.convert_tokens_to_ids("[/E1]")
        self.e2_open_id = tokenizer.convert_tokens_to_ids("[E2]")
        self.e2_close_id = tokenizer.convert_tokens_to_ids("[/E2]")

        # Class weights for imbalance
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None

    @staticmethod
    def _first_index(mask: torch.Tensor) -> torch.Tensor:
        """Find first True index in mask"""
        return mask.float().argmax(dim=1)

    def _span_mean(self, hidden: torch.Tensor, input_ids: torch.Tensor, open_id: int, close_id: int) -> torch.Tensor:
        """
        Mean-pool tokens strictly between open and close markers.
        
        This is the key innovation of this model - instead of using [CLS] token
        or attention mechanisms, we explicitly pool the representations of the
        entity spans to get focused entity representations.
        """
        B, L, H = hidden.shape
        pos = torch.arange(L, device=hidden.device).unsqueeze(0).expand(B, L)

        open_mask = (input_ids == open_id)
        close_mask = (input_ids == close_id)

        open_idx = self._first_index(open_mask)
        close_idx = self._first_index(close_mask)

        # span_mask[b, t] = True iff open_idx[b] < t < close_idx[b]
        span_mask = (pos > open_idx.unsqueeze(1)) & (pos < close_idx.unsqueeze(1))

        # Pool
        denom = span_mask.sum(dim=1, keepdim=True).clamp_min(1)
        span_sum = (hidden * span_mask.unsqueeze(-1)).sum(dim=1)
        span_mean = span_sum / denom

        # Fallback to open marker embedding if span empty
        has_tokens = span_mask.any(dim=1, keepdim=True)
        open_emb = (hidden * open_mask.unsqueeze(-1)).sum(dim=1)
        e_emb = torch.where(has_tokens, span_mean, open_emb)
        return e_emb

    def forward(self, input_ids, attention_mask=None, labels=None):
        """
        Forward pass of the model.
        
        Args:
            input_ids: Token IDs with entity markers [E1]...[/E1] and [E2]...[/E2]
            attention_mask: Attention mask for the input
            labels: Optional labels for training
            
        Returns:
            Dictionary with 'loss' and 'logits' keys
        """
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        last_hidden = outputs.last_hidden_state

        # Pool entity and date spans
        e1_emb = self._span_mean(last_hidden, input_ids, self.e1_open_id, self.e1_close_id)
        e2_emb = self._span_mean(last_hidden, input_ids, self.e2_open_id, self.e2_close_id)

        # Concatenate -> classify
        x = torch.cat([e1_emb, e2_emb], dim=-1)
        x = self.dropout(x)
        logits = self.classifier(x)

        loss = None
        if labels is not None:
            if hasattr(self, "class_weights") and self.class_weights is not None:
                loss_fn = nn.CrossEntropyLoss(weight=self.class_weights)
            else:
                loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)

        return {"loss": loss, "logits": logits}