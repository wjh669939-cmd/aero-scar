"""
Dual Stream Fusion Module

Supports multiple fusion strategies:
- 'concat': Concatenation fusion (default)
- 'residual_add': Residual addition fusion H = H_freq + H_time
- 'weighted_add': Weighted addition fusion H = α·H_freq + β·H_time (learnable weights)
- 'gated': Gated fusion H = g·H_freq + (1-g)·H_time (adaptive gating)
"""

import torch
import torch.nn as nn


class DualStreamFusion(nn.Module):
    """
    Dual-Stream Fusion Module.
    
    Fuses temporal representation (rep_T) and spectral representation (rep_F)
    into a unified representation Z.
    
    Supported fusion strategies:
    - 'concat': Concatenation (default, original approach)
    - 'residual_add': Residual addition H = H_freq (Trend) + H_time (Residual)
    - 'weighted_add': Weighted addition H = α·H_freq + β·H_time (learnable weights)
    - 'gated': Gated fusion H = g·H_freq + (1-g)·H_time (adaptive gating)
    
    Key design:
        - use_frequency parameter controls dual-stream usage, affects output_dim
        - When use_frequency=False (ablation Exp A), output_dim = rep_size
        - Ensures downstream modules (classifier, decoder) have consistent input dimensions
    """
    
    def __init__(self, rep_size, fusion_type='concat', dropout=0.1, use_frequency=True):
        super().__init__()
        self.rep_size = rep_size
        self.fusion_type = fusion_type
        self.use_frequency = use_frequency
        
        if fusion_type == 'concat':
            self.output_dim = rep_size * 2 if use_frequency else rep_size
            
        elif fusion_type == 'residual_add':
            self.output_dim = rep_size
            self.residual_proj = nn.Sequential(
                nn.LayerNorm(rep_size),
                nn.Linear(rep_size, rep_size),
                nn.Tanh()
            )
            
        elif fusion_type == 'weighted_add':
            self.output_dim = rep_size
            self.alpha = nn.Parameter(torch.tensor(0.5))
            self.beta = nn.Parameter(torch.tensor(0.5))
            self.norm = nn.LayerNorm(rep_size)
            
        elif fusion_type == 'gated':
            self.output_dim = rep_size
            self.gate = nn.Sequential(
                nn.Linear(rep_size * 2, rep_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(rep_size, rep_size),
                nn.Sigmoid()
            )
            self.norm = nn.LayerNorm(rep_size)
            
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")
    
    def forward(self, rep_T, rep_F=None):
        """
        Fuse dual-stream representations.
        
        Args:
            rep_T: (batch, rep_size) - Temporal representation (local details/residuals)
            rep_F: (batch, rep_size) - Spectral representation (global trends/seasonality, optional)
        
        Returns:
            fused: Fused representation
        """
        if rep_F is None:
            if self.fusion_type == 'concat':
                return rep_T
            else:
                return rep_T
        
        if self.fusion_type == 'concat':
            return torch.cat([rep_T, rep_F], dim=-1)
        
        elif self.fusion_type == 'residual_add':
            residual = self.residual_proj(rep_T)
            return rep_F + residual
        
        elif self.fusion_type == 'weighted_add':
            fused = self.alpha * rep_F + self.beta * rep_T
            return self.norm(fused)
        
        elif self.fusion_type == 'gated':
            gate_input = torch.cat([rep_T, rep_F], dim=-1)
            g = self.gate(gate_input)
            fused = g * rep_F + (1 - g) * rep_T
            return self.norm(fused)
    
    def get_fusion_weights(self, rep_T=None, rep_F=None):
        """
        Get fusion weights for visualization/analysis.
        
        Args:
            rep_T: Temporal representation (for gated fusion)
            rep_F: Spectral representation (for gated fusion)
        
        Returns:
            Dictionary of fusion weights
        """
        if self.fusion_type == 'weighted_add':
            return {
                'alpha (freq)': self.alpha.item(),
                'beta (time)': self.beta.item()
            }
        elif self.fusion_type == 'gated' and rep_T is not None and rep_F is not None:
            gate_input = torch.cat([rep_T, rep_F], dim=-1)
            g = self.gate(gate_input)
            return {
                'gate_mean': g.mean().item(),
                'gate_std': g.std().item()
            }
        return {}


class AttentionFusion(nn.Module):
    """
    Attention-based Fusion Module.
    
    Uses attention mechanism to fuse dual-stream representations.
    """
    def __init__(self, rep_size, dropout=0.1):
        super().__init__()
        self.output_dim = rep_size
        
        self.attention = nn.Sequential(
            nn.Linear(rep_size * 2, rep_size),
            nn.Tanh(),
            nn.Linear(rep_size, 1),
            nn.Softmax(dim=1)
        )
    
    def forward(self, rep_T, rep_F):
        """
        Fuse via attention mechanism.
        
        Args:
            rep_T: (B, rep_size) - Temporal representation
            rep_F: (B, rep_size) - Spectral representation
        
        Returns:
            fused: (B, rep_size) - Fused representation
        """
        combined = torch.cat([rep_T, rep_F], dim=-1)
        weights = self.attention(combined)
        
        fused = weights[:, :1] * rep_T + weights[:, 1:] * rep_F
        return fused
