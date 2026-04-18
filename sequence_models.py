import torch
import torch.nn as nn

class UnidirectionalGRU(nn.Module):
    """Unidirectional GRU model."""
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, dropout_rate):
        super(UnidirectionalGRU, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, text):
        embedded = self.embedding(text)
        _, hidden = self.gru(embedded)
        hidden = self.dropout(hidden.squeeze(0))
        output = self.fc(hidden)
        return output


class UnidirectionalLSTM(nn.Module):
    """Unidirectional LSTM model."""
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, dropout_rate):
        super(UnidirectionalLSTM, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, text):
        embedded = self.embedding(text)
        _, (hidden, cell) = self.lstm(embedded)
        hidden = self.dropout(hidden.squeeze(0))
        output = self.fc(hidden)
        return output


class BidirectionalGRU(nn.Module):
    """Bidirectional GRU model."""
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, dropout_rate):
        super(BidirectionalGRU, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, text):
        embedded = self.embedding(text)
        _, hidden = self.gru(embedded)
        hidden = torch.cat((hidden[0], hidden[1]), dim=1)
        hidden = self.dropout(hidden)
        output = self.fc(hidden)
        return output


class BidirectionalLSTM(nn.Module):
    """Bidirectional LSTM model."""
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, dropout_rate):
        super(BidirectionalLSTM, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, text):
        embedded = self.embedding(text)
        _, (hidden, cell) = self.lstm(embedded)
        hidden = torch.cat((hidden[0], hidden[1]), dim=1)
        hidden = self.dropout(hidden)
        output = self.fc(hidden)
        return output


# Model registry - maps model names to classes
MODELS = {
    'gru': UnidirectionalGRU,
    'gru_bi': BidirectionalGRU,
    'lstm': UnidirectionalLSTM,
    'lstm_bi': BidirectionalLSTM,
}


def get_model(model_name, vocab_size, embed_dim, hidden_dim, output_dim, dropout_rate):
    """
    Factory function to get a model by name.
    
    Args:
        model_name: One of 'gru', 'gru_bi', 'lstm', 'lstm_bi'
        vocab_size: Vocabulary size
        embed_dim: Embedding dimension
        hidden_dim: Hidden dimension
        output_dim: Output dimension (1 for binary classification)
        dropout_rate: Dropout rate
        
    Returns:
        Model instance
    """
    if model_name not in MODELS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODELS.keys())}")
    
    model_class = MODELS[model_name]
    return model_class(vocab_size, embed_dim, hidden_dim, output_dim, dropout_rate)