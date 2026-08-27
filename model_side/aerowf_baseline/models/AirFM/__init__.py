"""
AirFM Models Package

Recommended usage:
    from models.AirFM import UnifiedSeries2Vec
    from models.AirFM import UnifiedTrainer

Module structure:
    - encoders/: Encoder modules
        - transformer_encoder.py: Transformer Temporal Encoder
        - frets_encoder.py: FreTS Spectral (Frequency-domain) Encoder
        - positional_encoding.py: Position Encoding
        - exogenous_encoder.py: Exogenous Variable Encoder
    - fusion/: Fusion modules
        - dual_stream_fusion.py: Dual-stream Fusion
    - HierarchicalGNN.py: Hierarchical Aggregator
    - masked.py: Masked Reconstruction utilities
    - unified_model.py: Unified Model
    - unified_trainer.py: Unified Trainer
"""

from .encoders import (
    TransformerEncoder,
    FreTSEncoder,
    ExogenousEncoder,
    PositionalEncoding,
)

from .fusion import (
    DualStreamFusion,
    AttentionFusion,
)

from .HierarchicalGNN import (
    HierarchicalGNN,
    SimpleHierarchicalGNN,
)

from .masked import (
    generate_hybrid_mask,
    apply_mask,
    masked_mse_loss,
    ReconstructionDecoder
)

from .unified_model import UnifiedSeries2Vec
from .unified_trainer import UnifiedTrainer

__all__ = [
    # Encoders
    'TransformerEncoder',
    'FreTSEncoder',
    'ExogenousEncoder',
    'PositionalEncoding',
    
    # Fusion
    'DualStreamFusion',
    'AttentionFusion',
    
    # Hierarchical Aggregator
    'HierarchicalGNN',
    'SimpleHierarchicalGNN',
    
    # Masked Reconstruction utilities
    'generate_hybrid_mask',
    'apply_mask',
    'masked_mse_loss',
    'ReconstructionDecoder',
    
    # Unified Model and Trainer
    'UnifiedSeries2Vec',
    'UnifiedTrainer',
]
