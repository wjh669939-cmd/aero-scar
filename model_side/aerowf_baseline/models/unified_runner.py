"""
Unified Training Runner for AeroWF

Model architecture:
    Input → Dual-stream Encoders (Temporal/Spectral) → Hierarchical Aggregator → 
    Representation → Pre-training Tasks

Data formats:
    - Multi-node format (Multi-runway airports): (batch, num_nodes, time_steps, channels)
    - Single-node format: (batch, channels, time_steps)

Usage:
    from models.unified_runner import unified_train
    results = unified_train(config, Data)
"""

import os
import logging
import torch
from torch.utils.data import DataLoader, TensorDataset

from models.AirFM.unified_model import UnifiedSeries2Vec
from models.AirFM.unified_trainer import UnifiedTrainer

logger = logging.getLogger('__main__')


def unified_train(config, Data):
    """
    Unified training entry point for AeroWF model.
    
    Args:
        config: Configuration dictionary
        Data: Data dictionary (loaded from dataloader)
    
    Returns:
        best_metrics: Best model metrics
        all_metrics: All training metrics
    """
    logger.info("=" * 70)
    logger.info("Unified Training Entry - AeroWF Training Runner")
    logger.info("=" * 70)
    
    # Set task type
    task_type = config.get('task_type', 'masked_recon')
    config['task_type'] = task_type
    config['training_mode'] = task_type
    
    logger.info(f"Task type: {task_type}")
    
    # Determine data format and load data
    if isinstance(Data, tuple) and len(Data) == 3:
        # New format: (train_loader, val_loader, test_loader) - Multi-airport mode
        train_loader, val_loader, test_loader = Data
        # Get sample from DataLoader to determine data shape
        sample_batch = next(iter(train_loader))
        if len(sample_batch) >= 2:
            X_sample = sample_batch[0]
            train_data = X_sample
            train_label = sample_batch[1]
        else:
            raise ValueError("DataLoader returned incorrect data format")
        
        train_data = train_data.cpu().numpy() if hasattr(train_data, 'cpu') else train_data
        train_label = train_label.cpu().numpy() if hasattr(train_label, 'cpu') else train_label
        
        val_data = None
        val_label = None
        test_data = None
        test_label = None
        logger.info("Data format: Multi-airport DataLoader (tuple)")
    else:
        # Old format: Data is dict {'train_data': ..., 'train_label': ...}
        train_data = Data['train_data']
        train_label = Data['train_label']
        val_data = Data.get('val_data', train_data)
        val_label = Data.get('val_label', train_label)
        test_data = Data['test_data']
        test_label = Data['test_label']
        logger.info("Data format: Dictionary format")
    
    # Determine data format
    data_shape = train_data.shape
    is_multi_node = len(data_shape) == 4
    
    if is_multi_node:
        # Multi-node format: (batch, num_nodes, time_steps, channels)
        _, num_nodes, time_steps, channels = data_shape
        logger.info(f"Multi-node data format: {data_shape}")
        logger.info(f"  - Number of nodes: {num_nodes}")
        logger.info(f"  - Time steps: {time_steps}")
        logger.info(f"  - Number of features: {channels}")
        config['num_nodes'] = num_nodes
    else:
        # Single-node format: (batch, channels, time_steps)
        _, channels, time_steps = data_shape
        logger.info(f"Single-node data format: {data_shape}")
        logger.info(f"  - Number of channels: {channels}")
        logger.info(f"  - Time steps: {time_steps}")
    
    # Set model configuration
    config['Data_shape'] = data_shape
    config['num_labels'] = int(max(train_label)) + 1
    logger.info(f"Number of classes: {config['num_labels']}")
    
    # Create model
    logger.info("Creating AeroWF model...")
    model = UnifiedSeries2Vec(config, num_classes=config['num_labels'])
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {num_params:,}")
    
    # Check if already in DataLoader format
    if isinstance(Data, tuple) and len(Data) == 3:
        # Multi-airport mode: already (train_loader, val_loader, test_loader)
        logger.info("Using multi-airport DataLoader format")
        # Validation and test loaders may be None
        _, val_loader, test_loader = Data
        train_loader = Data[0]
    else:
        # Old format: create DataLoader
        train_loader, val_loader, test_loader = _create_data_loaders(
            train_data, train_label,
            val_data, val_label,
            test_data, test_label,
            config
        )
    
    # Create trainer
    device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    trainer = UnifiedTrainer(model, config, device=device)
    
    # Start training
    logger.info("Starting training...")
    all_metrics = trainer.train(
        train_loader, val_loader, test_loader,
        num_epochs=config.get('epochs', 100)
    )
    
    # Evaluate after pre-training
    if task_type == 'masked_recon':
        # Masked reconstruction task: evaluate reconstruction quality on test set
        # Simplified: h_F (Trend) + h_T (Residual) → decoder → temporal reconstruction
        logger.info("\n" + "=" * 70)
        logger.info("Evaluating masked reconstruction quality on test set...")
        logger.info("  Data flow: x_masked → [encoder_T, encoder_F] → h_T + h_F → decoder → reconstruction")
        logger.info("=" * 70)
        
        recon_metrics = trainer.evaluate_reconstruction(test_loader)
        all_metrics.update(recon_metrics)
        
        logger.info(f"Test set reconstruction loss: {recon_metrics.get('test_recon_loss', 0):.4f}")
        if 'recon_mse' in recon_metrics:
            logger.info(f"Reconstruction MSE: {recon_metrics.get('recon_mse', 0):.4f}")
        
    elif task_type == 'pretrain':
        # Contrastive learning pre-training: perform linear probing evaluation
        logger.info("\n" + "=" * 70)
        logger.info("Evaluating downstream tasks (linear probing)...")
        logger.info("=" * 70)
        
        lp_metrics = trainer.evaluate_linear_probe(test_loader, train_loader)
        all_metrics['linear_probe_accuracy'] = lp_metrics['accuracy']
        all_metrics['linear_probe_f1'] = lp_metrics['f1_macro']
        
        logger.info(f"Linear probe accuracy: {lp_metrics['accuracy']:.4f}")
        logger.info(f"Linear probe F1: {lp_metrics['f1_macro']:.4f}")
        
        # Optional: fine-tuning
        if config.get('do_finetune', True):
            logger.info("\n" + "=" * 70)
            logger.info("Performing fine-tuning...")
            logger.info("=" * 70)
            
            finetune_metrics = _finetune(model, train_loader, val_loader, test_loader, config, device)
            all_metrics.update({f'finetune_{k}': v for k, v in finetune_metrics.items()})
    
    # Extract best metrics
    best_metrics = _extract_best_metrics(all_metrics, task_type)
    
    # Save results
    _save_results(config, all_metrics)
    
    logger.info("=" * 70)
    logger.info("Training completed!")
    logger.info(f"Best metrics: {best_metrics}")
    logger.info("=" * 70)
    
    return best_metrics, all_metrics


def _create_data_loaders(train_data, train_label, val_data, val_label,
                         test_data, test_label, config):
    """Create data loaders from tensors."""
    batch_size = config.get('batch_size', 64)
    num_workers = config.get('num_workers', 0)
    
    # Convert to tensor
    def to_tensor(data, label):
        if not isinstance(data, torch.Tensor):
            data = torch.FloatTensor(data)
        if not isinstance(label, torch.Tensor):
            label = torch.LongTensor(label)
        return data, label
    
    train_data, train_label = to_tensor(train_data, train_label)
    val_data, val_label = to_tensor(val_data, val_label)
    test_data, test_label = to_tensor(test_data, test_label)
    
    # Create datasets
    train_dataset = TensorDataset(train_data, train_label, torch.arange(len(train_data)))
    val_dataset = TensorDataset(val_data, val_label, torch.arange(len(val_data)))
    test_dataset = TensorDataset(test_data, test_label, torch.arange(len(test_data)))
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                              pin_memory=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                           pin_memory=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                            pin_memory=True, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader


def _finetune(model, train_loader, val_loader, test_loader, config, device):
    """Fine-tune the pre-trained AeroWF model."""
    finetune_config = config.copy()
    finetune_config['training_mode'] = 'supervised'
    finetune_config['epochs'] = config.get('finetune_epochs', 50)
    finetune_config['lr'] = config.get('finetune_lr', 1e-4)
    
    # Unfreeze encoder
    model.unfreeze_encoder()
    
    trainer = UnifiedTrainer(model, finetune_config, device=device)
    metrics = trainer.train(train_loader, val_loader, test_loader,
                           num_epochs=finetune_config['epochs'])
    
    return metrics


def _extract_best_metrics(all_metrics, task_type):
    """Extract best metrics based on task type."""
    best_metrics = {}
    
    if task_type == 'supervised':
        best_metrics['accuracy'] = all_metrics.get('test_accuracy', 
                                   all_metrics.get('val_accuracy', 0))
        best_metrics['f1'] = all_metrics.get('test_f1_macro',
                            all_metrics.get('val_f1', 0))
    
    elif task_type == 'masked_recon':
        # Masked reconstruction task key metrics (simplified: temporal reconstruction only)
        best_metrics['test_recon_loss'] = all_metrics.get('test_recon_loss', 0)
        if 'recon_mse' in all_metrics:
            best_metrics['recon_mse'] = all_metrics['recon_mse']
    
    elif task_type == 'pretrain':
        # Contrastive learning pre-training metrics
        if 'linear_probe_accuracy' in all_metrics:
            best_metrics['linear_probe_accuracy'] = all_metrics['linear_probe_accuracy']
        if 'finetune_test_accuracy' in all_metrics:
            best_metrics['finetune_accuracy'] = all_metrics['finetune_test_accuracy']
    
    best_metrics['train_loss'] = all_metrics.get('train_loss', 0)
    best_metrics['val_loss'] = all_metrics.get('val_loss', 0)
    
    return best_metrics


def _save_results(config, metrics):
    """Save training results to file."""
    output_dir = config.get('output_dir', './Results')
    problem = config.get('problem', 'unknown')
    
    os.makedirs(output_dir, exist_ok=True)
    
    results_file = os.path.join(output_dir, f'{problem}_results.txt')
    
    with open(results_file, 'w') as f:
        f.write(f"Problem: {problem}\n")
        f.write(f"Task Type: {config.get('task_type', 'unknown')}\n")
        f.write("-" * 50 + "\n")
        for key, value in metrics.items():
            if isinstance(value, float):
                f.write(f"{key}: {value:.6f}\n")
            else:
                f.write(f"{key}: {value}\n")
    
    logger.info(f"Results saved to: {results_file}")
