"""
Model Factory - Create models based on configuration.

Supported models:
    - UnifiedSeries2Vec (AeroWF - Recommended)
"""

import logging
from models.AirFM.unified_model import UnifiedSeries2Vec


logger = logging.getLogger('__main__')


def count_parameters(model):
    """Count trainable model parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def Model_factory(config, data):
    """
    Model factory function to create models.
    
    Args:
        config: Configuration dictionary
        data: Data dictionary
    
    Returns:
        model: Created model instance
    """
    config['Data_shape'] = data['train_data'].shape
    config['num_labels'] = int(max(data['train_label'])) + 1

    if config['Model_Type'][0] == 'AirFM':
            logger.info("Creating Unified AeroWF Model...")
            model = UnifiedSeries2Vec(config, num_classes=config['num_labels'])
    else:
        raise ValueError(f"Unknown model type: {config['Model_Type'][0]}")

    logger.info("Model:\n{}".format(model))
    logger.info("Total number of parameters: {:,}".format(count_parameters(model)))
    return model


def create_unified_model(config, data):
    """
    Convenience function to create the unified AeroWF model.
    
    Args:
        config: Configuration dictionary
        data: Data dictionary
    
    Returns:
        model: UnifiedSeries2Vec (AeroWF) model instance
    """
    config['Data_shape'] = data['train_data'].shape
    config['num_labels'] = int(max(data['train_label'])) + 1
    
    model = UnifiedSeries2Vec(config, num_classes=config['num_labels'])
    
    logger.info(f"Created AeroWF model with {count_parameters(model):,} parameters")
    return model
