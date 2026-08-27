"""
Transformer Encoder Module - Temporal Branch

Transformer-based encoder for capturing long-range dependencies in time series.
"""

import torch
import torch.nn as nn
from .positional_encoding import PositionalEncoding, LearnablePositionalEncoding


class TransformerEncoder(nn.Module):
    """
    Transformer-based Temporal Encoder for Time Series.
    
    Captures long-range dependencies and temporal patterns.
    Supports variable-length sequences and configurable network depth/width.
    
    Input: (batch, channels, time_steps)
    Output: (batch, rep_size, reduced_time)
    """
    
    def __init__(
        self,
        channel_size,
        emb_size,
        rep_size,
        num_heads=4,
        num_layers=2,
        dim_ff=256,
        dropout=0.1,
        kernel_size=8,
        use_learnable_pe=False
    ):
        super().__init__()
        
        self.channel_size = channel_size
        self.emb_size = emb_size
        self.rep_size = rep_size
        
        self.input_proj = nn.Linear(channel_size, emb_size)
        
        if use_learnable_pe:
            self.pos_encoder = LearnablePositionalEncoding(
                max_len=5000, d_model=emb_size, dropout=dropout
            )
        else:
            self.pos_encoder = PositionalEncoding(
                d_model=emb_size, dropout=dropout
            )
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_size,
            nhead=num_heads,
            dim_feedforward=dim_ff,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )
        
        self.output_proj = nn.Sequential(
            nn.Linear(emb_size, rep_size),
            nn.GELU()
        )
        
        self.downsample = nn.Sequential(
            nn.Conv1d(rep_size, rep_size, kernel_size=kernel_size, padding='valid'),
            nn.BatchNorm1d(rep_size),
            nn.GELU(),
            nn.Conv1d(rep_size, rep_size, kernel_size=3, padding='valid'),
            nn.BatchNorm1d(rep_size),
            nn.GELU()
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x):
        """
        Forward pass for Transformer encoder.
        
        Args:
            x: (batch, channels, time_steps)
        
        Returns:
            output: (batch, rep_size, reduced_time)
        """
        B, C, T = x.shape
        
        x = x.permute(0, 2, 1)
        
        x = self.input_proj(x)
        
        x = self.pos_encoder(x)
        
        x = self.transformer_encoder(x)
        
        x = self.output_proj(x)
        
        x = x.permute(0, 2, 1)
        
        x = self.downsample(x)
        
        return x


class LightweightTransformerEncoder(nn.Module):
    """
    Lightweight Transformer Encoder for resource-constrained settings.
    
    Simplified architecture with fewer layers and reduced dimensions.
    """
    
    def __init__(
        self,
        channel_size,
        emb_size,
        rep_size,
        num_heads=4,
        num_layers=1,
        dim_ff=128,
        dropout=0.1
    ):
        super().__init__()
        
        self.input_proj = nn.Linear(channel_size, emb_size)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_size,
            nhead=num_heads,
            dim_feedforward=dim_ff,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )
        
        self.output_proj = nn.Linear(emb_size, rep_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Forward pass for lightweight Transformer encoder.
        
        Args:
            x: (batch, channels, time_steps)
        
        Returns:
            output: (batch, rep_size, time)
        """
        B, C, T = x.shape
        
        x = x.permute(0, 2, 1)
        x = self.input_proj(x)
        x = self.transformer_encoder(x)
        x = self.output_proj(x)
        
        return x.permute(0, 2, 1)
