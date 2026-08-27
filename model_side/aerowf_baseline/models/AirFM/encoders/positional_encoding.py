"""
Positional Encoding Module - for Transformer

Encodes position information in sequences using sinusoidal functions.
"""

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding.
    
    Encodes position information using sine and cosine functions of varying frequencies.
    Generalizes to arbitrary sequence lengths and preserves relative position information.
    """
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            x + positional encoding
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class LearnablePositionalEncoding(nn.Module):
    """
    Learnable Positional Encoding.
    
    Used when sequence length is relatively fixed. Parameters are learned during training.
    """
    def __init__(self, max_len, d_model, dropout=0.1):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, d_model))
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            x + learnable positional encoding
        """
        seq_len = x.size(1)
        pos_embed = self.pos_embed[:, :seq_len, :]
        return self.dropout(x + pos_embed)
