"""
Exogenous Encoder Module - METAR Integration

Encodes meteorological variables (METAR) and fuses them with sensor observations.

Design:
- Categorical variables: Embedding layer
- Continuous variables: MLP projection
- Fusion: Additive fusion to preserve input dimensions
"""

import torch
import torch.nn as nn


class ExogenousEncoder(nn.Module):
    """
    Exogenous variable encoder for METAR meteorological data.
    
    Fuses external meteorological information with sensor observations.
    
    Args:
        exo_config: Configuration dictionary for exogenous variables
            {
                'categorical': {
                    'weather_code_id': {'vocab_size': 25},
                    'sky_condition': {'vocab_size': 6},
                    ...
                },
                'continuous': ['visibility', 'cloud_height']
            }
        input_channels: Number of sensor channels
        seq_len: Sequence length
        dropout: Dropout probability
    """
    
    def __init__(self, exo_config, input_channels, seq_len, dropout=0.1):
        super().__init__()
        
        self.exo_config = exo_config
        self.input_channels = input_channels
        self.seq_len = seq_len
        
        embed_dim = input_channels * 2
        
        self.cat_embeddings = nn.ModuleDict()
        cat_config = exo_config.get('categorical', {})
        for name, cfg in cat_config.items():
            vocab_size = cfg['vocab_size']
            self.cat_embeddings[name] = nn.Embedding(
                num_embeddings=vocab_size,
                embedding_dim=embed_dim,
                padding_idx=0
            )
        
        cont_vars = exo_config.get('continuous', [])
        self.num_continuous = len(cont_vars)
        if self.num_continuous > 0:
            self.cont_proj = nn.Sequential(
                nn.Linear(self.num_continuous, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim, embed_dim)
            )
        
        self.to_input_space = nn.Sequential(
            nn.Linear(embed_dim, input_channels * seq_len),
            nn.LayerNorm(input_channels * seq_len),
            nn.Dropout(dropout)
        )
        
        self.exo_scale = nn.Parameter(torch.ones(1) * 0.1)
    
    def forward(self, exo_categorical=None, exo_continuous=None):
        """
        Encode exogenous variables.
        
        Args:
            exo_categorical: Dictionary of categorical variables {'name': (B,)}
            exo_continuous: Dictionary of continuous variables {'name': (B,)}
        
        Returns:
            exo_embed: (B, input_channels, seq_len) or None
        """
        embeddings = []
        batch_size = None
        
        if exo_categorical is not None:
            for name, emb_layer in self.cat_embeddings.items():
                if name in exo_categorical:
                    ids = exo_categorical[name]
                    if batch_size is None:
                        batch_size = ids.shape[0]
                    emb = emb_layer(ids)
                    embeddings.append(emb)
        
        if exo_continuous is not None and self.num_continuous > 0:
            cont_names = self.exo_config.get('continuous', [])
            cont_values = []
            for name in cont_names:
                if name in exo_continuous:
                    val = exo_continuous[name].unsqueeze(-1)
                    if batch_size is None:
                        batch_size = val.shape[0]
                    cont_values.append(val)
            
            if cont_values:
                cont_tensor = torch.cat(cont_values, dim=-1)
                cont_emb = self.cont_proj(cont_tensor)
                embeddings.append(cont_emb)
        
        if embeddings:
            exo_embed = torch.stack(embeddings, dim=0).sum(dim=0)
            exo_input = self.to_input_space(exo_embed)
            exo_input = exo_input.view(batch_size, self.input_channels, self.seq_len)
            exo_input = self.exo_scale * exo_input
            return exo_input
        else:
            return None
    
    def get_info(self):
        """Get exogenous encoder configuration info."""
        return {
            'input_channels': self.input_channels,
            'seq_len': self.seq_len,
            'categorical_vars': list(self.cat_embeddings.keys()),
            'num_continuous': self.num_continuous,
            'exo_scale': self.exo_scale.item()
        }
