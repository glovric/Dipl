import torch
import torch.nn as nn
import math

# Scaled Dot-Product Attention
class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        output = torch.matmul(attn, value)
        return output, attn


# Feed Forward Network
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))


# Encoder Layer
class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.attn = ScaledDotProductAttention(dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        # Attention layer
        attn_output, _ = self.attn(x, x, x, mask)
        x = self.layer_norm1(x + attn_output)
        
        # Feed-forward layer
        ff_output = self.ffn(x)
        x = self.layer_norm2(x + ff_output)
        
        return x


# Encoder
class Encoder(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, n_layers, dropout=0.1):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x


# Sequence-to-Vector Transformer for Financial Data
class FinancialTransformer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, n_enc_layers, n_features, dropout=0.1):
        super(FinancialTransformer, self).__init__()
        self.encoder = Encoder(d_model, n_heads, d_ff, n_enc_layers, dropout)
        self.fc_out = nn.Linear(d_model, n_features)  # Output layer to predict n_features
        
    def forward(self, x, mask=None):
        # Pass input through encoder
        enc_output = self.encoder(x, mask)
        
        # Apply Global Average Pooling across sequence length
        pooled_output = torch.mean(enc_output, dim=1)  # Shape: [batch_size, d_model]
        
        # Output layer to produce a fixed vector of shape [batch_size, n_features]
        output = self.fc_out(pooled_output)
        
        return output