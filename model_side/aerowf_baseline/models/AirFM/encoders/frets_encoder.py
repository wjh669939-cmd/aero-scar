"""
FreTS Encoder Module - Spectral Branch

Frequency-domain time series encoder based on FreTS (Frequency-domain MLPs).

Reference: "Frequency-domain MLPs are More Effective Learners in Time Series Forecasting"

Architecture:
  FFT → Frequency Learning (Channel+Temporal MLPs) → iFFT → Output Projection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ComplexMLP(nn.Module):
    """
    Complex-valued MLP: processes real and imaginary parts independently.
    
    Preserves phase information while learning complex transformations.
    """
    def __init__(self, in_features, out_features, dropout=0.1):
        super().__init__()
        self.mlp_real = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.mlp_imag = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.GELU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, x_real, x_imag):
        """
        Args:
            x_real: Real part (...)
            x_imag: Imaginary part (...)
        Returns:
            out_real, out_imag: Processed real and imaginary parts
        """
        return self.mlp_real(x_real), self.mlp_imag(x_imag)


class FrequencyChannelLearner(nn.Module):
    """
    Frequency Channel Learner.
    
    Captures multi-variable correlations in frequency domain.
    Operates on channel dimension with independent MLP for real and imaginary parts.
    
    Input: (B, C, F) - Batch, Channel, Frequency
    Output: (B, C, F)
    """
    def __init__(self, num_channels, hidden_dim=None, dropout=0.1):
        super().__init__()
        hidden_dim = hidden_dim or num_channels * 2
        
        self.channel_mlp_real = nn.Sequential(
            nn.Linear(num_channels, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_channels),
            nn.Dropout(dropout)
        )
        self.channel_mlp_imag = nn.Sequential(
            nn.Linear(num_channels, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_channels),
            nn.Dropout(dropout)
        )
        
        self.norm_real = nn.LayerNorm(num_channels)
        self.norm_imag = nn.LayerNorm(num_channels)
    
    def forward(self, x_real, x_imag):
        """
        Args:
            x_real: (B, C, F) Real part
            x_imag: (B, C, F) Imaginary part
        Returns:
            out_real, out_imag: (B, C, F)
        """
        B, C, F = x_real.shape
        
        # Transpose to (B, F, C) for MLP on channel dimension
        x_real_t = x_real.permute(0, 2, 1)
        x_imag_t = x_imag.permute(0, 2, 1)
        
        # MLP + residual connection
        out_real = self.channel_mlp_real(x_real_t)
        out_imag = self.channel_mlp_imag(x_imag_t)
        
        # Layer normalization
        out_real = self.norm_real(out_real + x_real_t)
        out_imag = self.norm_imag(out_imag + x_imag_t)
        
        return out_real.permute(0, 2, 1), out_imag.permute(0, 2, 1)


class FrequencyTemporalLearner(nn.Module):
    """
    Frequency Temporal Learner.
    
    Captures global temporal patterns (periodicity, trends) in frequency domain.
    Operates on frequency dimension with independent MLP for real and imaginary parts.
    
    Input: (B, C, F) - Batch, Channel, Frequency
    Output: (B, C, F)
    """
    def __init__(self, freq_bins, hidden_dim=None, dropout=0.1):
        super().__init__()
        hidden_dim = hidden_dim or freq_bins * 2
        
        # MLPs on frequency dimension (handle real and imaginary parts separately)
        self.freq_mlp_real = nn.Sequential(
            nn.Linear(freq_bins, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, freq_bins),
            nn.Dropout(dropout)
        )
        self.freq_mlp_imag = nn.Sequential(
            nn.Linear(freq_bins, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, freq_bins),
            nn.Dropout(dropout)
        )
        
        # Layer normalization
        self.norm_real = nn.LayerNorm(freq_bins)
        self.norm_imag = nn.LayerNorm(freq_bins)
    
    def forward(self, x_real, x_imag):
        """
        Args:
            x_real: (B, C, F) Real part
            x_imag: (B, C, F) Imaginary part
        Returns:
            out_real, out_imag: (B, C, F)
        """
        # Apply MLPs directly on frequency dimension F
        out_real = self.freq_mlp_real(x_real)
        out_imag = self.freq_mlp_imag(x_imag)
        
        # Layer normalization + residual connection
        out_real = self.norm_real(out_real + x_real)
        out_imag = self.norm_imag(out_imag + x_imag)
        
        return out_real, out_imag


class FreTSBlock(nn.Module):
    """
    FreTS Block: combines frequency channel learner and frequency temporal learner.
    
    Each block contains:
    1. Frequency Channel Learner - learns variable correlations
    2. Frequency Temporal Learner - learns temporal patterns
    """
    def __init__(self, num_channels, freq_bins, hidden_dim=None, dropout=0.1):
        super().__init__()
        
        self.channel_learner = FrequencyChannelLearner(
            num_channels, hidden_dim=hidden_dim, dropout=dropout
        )
        self.temporal_learner = FrequencyTemporalLearner(
            freq_bins, hidden_dim=hidden_dim, dropout=dropout
        )
    
    def forward(self, x_real, x_imag):
        """
        Args:
            x_real, x_imag: (B, C, F)
        Returns:
            out_real, out_imag: (B, C, F)
        """
        # Frequency channel learning
        x_real, x_imag = self.channel_learner(x_real, x_imag)
        
        # Frequency temporal learning
        x_real, x_imag = self.temporal_learner(x_real, x_imag)
        
        return x_real, x_imag


class FreTSEncoder(nn.Module):
    """
    Frequency-domain encoder based on FreTS paper.
    
    Complete architecture:
        Domain Transform (FFT) → Frequency Learning (Channel+Temporal MLP stack) → 
        Domain Inverse Transform (iFFT) → Output Projection
    
    Input: (batch, channels, time_steps) temporal data
    Output: (batch, rep_size, reduced_time) temporal data (converted back)
    
    Key Design:
        - seq_len is fixed in __init__, freq_bins = seq_len // 2 + 1
        - FreTS blocks created in __init__, not dynamically in forward (prevents weight loss)
        - If input length != seq_len, automatically interpolates for alignment
    """
    def __init__(
        self,
        channel_size,
        emb_size,
        rep_size,
        seq_len=96,  # Fixed: sequence length
        num_layers=2,
        dropout=0.1,
        kernel_size=8
    ):
        super().__init__()
        
        self.channel_size = channel_size
        self.emb_size = emb_size
        self.rep_size = rep_size
        self.num_layers = num_layers
        
        # ========== Key Fix: Fixed sequence length and frequency dimension ==========
        # Determined in __init__, not dynamically created in forward
        self.seq_len = seq_len
        self.freq_bins = seq_len // 2 + 1  # Frequency dimension from rfft output
        
        # Input projection: expand channel dimension to emb_size
        self.input_proj_real = nn.Linear(channel_size, emb_size)
        self.input_proj_imag = nn.Linear(channel_size, emb_size)
        
        # ========== Key Fix: Create FreTS Blocks in __init__ ==========
        # Prevents dynamic creation in forward from losing weights in Optimizer tracking
        self.frets_blocks = nn.ModuleList([
            FreTSBlock(
                num_channels=emb_size,
                freq_bins=self.freq_bins,
                hidden_dim=emb_size * 2,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        
        # Output projection: convert frequency-learned features to rep_size
        self.output_proj = nn.Sequential(
            nn.Linear(emb_size, rep_size),
            nn.GELU()
        )
        
        # Downsampling convolution
        self.downsample = nn.Sequential(
            nn.Conv1d(rep_size, rep_size, kernel_size=kernel_size, padding='valid'),
            nn.BatchNorm1d(rep_size),
            nn.GELU(),
            nn.Conv1d(rep_size, rep_size, kernel_size=3, padding='valid'),
            nn.BatchNorm1d(rep_size),
            nn.GELU()
        )
        
        self.dropout = nn.Dropout(dropout)
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """
        Args:
            x: (batch, channels, time_steps) temporal data
        
        Returns:
            output: (batch, rep_size, reduced_time)
        
        Note:
            If input time_steps != self.seq_len, automatically interpolates to fixed length,
            then interpolates back to original length. Ensures frequency domain MLP weight dimensions always match.
        """
        B, C, T = x.shape
        original_T = T  # Store original length
        
        # ========== Input length alignment: interpolate to fixed seq_len ==========
        if T != self.seq_len:
            # Use linear interpolation to scale input to fixed length
            x = F.interpolate(x, size=self.seq_len, mode='linear', align_corners=False)
            T = self.seq_len
        
        # FFT transform to frequency domain
        x_fft = torch.fft.rfft(x, dim=-1)  # (B, C, freq_bins)
        
        # ===== Stage 1: Domain Transform - Separate real and imaginary parts =====
        x_real = x_fft.real  # (B, C, F)
        x_imag = x_fft.imag  # (B, C, F)
        
        # ===== Stage 2: Input Projection =====
        # Expand channel dimension: (B, C, F) -> (B, F, C) -> MLP -> (B, F, emb_size) -> (B, emb_size, F)
        x_real_t = x_real.permute(0, 2, 1)
        x_imag_t = x_imag.permute(0, 2, 1)
        
        x_real_proj = self.input_proj_real(x_real_t).permute(0, 2, 1)
        x_imag_proj = self.input_proj_imag(x_imag_t).permute(0, 2, 1)
        
        # ===== Stage 3: Frequency Domain Learning - Stack FreTS Blocks =====
        for block in self.frets_blocks:
            x_real_proj, x_imag_proj = block(x_real_proj, x_imag_proj)
        
        # ===== Stage 4: Domain Inverse Transform - Reconstruct complex and iFFT =====
        x_complex = torch.complex(x_real_proj, x_imag_proj)
        x_time = torch.fft.irfft(x_complex, n=T, dim=-1)  # (B, emb_size, T)
        
        # ===== Stage 5: Output Projection ====
        x_out = x_time.permute(0, 2, 1)  # (B, T, emb_size)
        x_out = self.output_proj(x_out)  # (B, T, rep_size)
        x_out = x_out.permute(0, 2, 1)   # (B, rep_size, T)
        
        # ========== Output length recovery: interpolate back to original length if needed ==========
        # Note: interpolate back before downsampling to ensure consistent downstream processing
        if original_T != self.seq_len:
            x_out = F.interpolate(x_out, size=original_T, mode='linear', align_corners=False)
        
        # Downsampling
        x_out = self.downsample(x_out)
        
        return x_out
