"""
AeroWF: Geometric-Aware Spectral-Temporal Dual-Stream Learning for Airport Weather Forecasting

Main training entry point supporting:
  - Unified pre-training (masked reconstruction + contrastive learning)
  - Single-airport training
  - Multi-airport training with variable runway cardinality

Usage:
  python main.py --task_type unified_pretrain --lambda_recon 1.0 --lambda_contrast 0.5
  python main.py --task_type masked_recon
  python main.py --use_multi_airport --N_max 4
"""

import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from utils import args
from models.unified_runner import unified_train
from Dataset.multi_airport_dataset import create_multi_airport_loaders


def load_single_airport_data(config):
    """
    Load single-airport data from .npy files.
    
    Expected directory structure:
      data_dir/
        {problem}/
          {problem}_train.npy (train_data, train_label)
          {problem}_val.npy   (val_data, val_label)
          {problem}_test.npy  (test_data, test_label)
    """
    problem = config['problem']
    data_dir = config['data_dir']
    
    def load_split(split_name):
        filepath = os.path.join(data_dir, problem, f'{problem}_{split_name}.npy')
        if os.path.exists(filepath):
            data = np.load(filepath, allow_pickle=True)
            if isinstance(data, dict):
                X = data.get('data', data.get('train_data'))
                y = data.get('label', data.get('train_label'))
            else:
                X, y = data[0], data[1]
            return X, y
        return None, None
    
    train_X, train_y = load_split('train')
    val_X, val_y = load_split('val')
    test_X, test_y = load_split('test')
    
    if train_X is None:
        raise FileNotFoundError(f"Training data not found: {os.path.join(data_dir, problem, f'{problem}_train.npy')}")
    
    class SimpleDataset(torch.utils.data.Dataset):
        def __init__(self, X, y):
            self.X = torch.from_numpy(X).float() if X is not None else None
            self.y = torch.from_numpy(y).long() if y is not None else None
        
        def __len__(self):
            return len(self.y) if self.y is not None else 0
        
        def __getitem__(self, idx):
            if self.X is not None:
                return self.X[idx], self.y[idx]
            return self.y[idx]
    
    train_dataset = SimpleDataset(train_X, train_y)
    val_dataset = SimpleDataset(val_X, val_y) if val_X is not None else None
    test_dataset = SimpleDataset(test_X, test_y) if test_X is not None else None
    
    train_loader = DataLoader(train_dataset, batch_size=config.get('batch_size', 32), shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.get('batch_size', 32), shuffle=False) if val_dataset else None
    test_loader = DataLoader(test_dataset, batch_size=config.get('batch_size', 32), shuffle=False) if test_dataset else None
    
    return train_loader, val_loader, test_loader


def main():
    config = args.Initialization(args)
    use_multi_airport = config.get('use_multi_airport', False)
    
    if use_multi_airport:
        print(f"\n{'='*70}")
        print("Mode: Multi-Airport Joint Training")
        print(f"{'='*70}\n")
        
        train_loader, val_loader, test_loader = create_multi_airport_loaders(
            data_dir=config.get('data_dir', 'Dataset/processed'),
            batch_size=config.get('batch_size', 8),
            N_max=config.get('N_max', 4),
            mix_ratio=config.get('mix_ratio', 'uniform'),
            num_workers=config.get('num_workers', 0)
        )
        
        Data = (train_loader, val_loader, test_loader)
        best_aggr_metrics_test, all_metrics = unified_train(config, Data)
        
    else:
        problems = [
            p for p in os.listdir(config['data_dir']) 
            if os.path.isdir(os.path.join(config['data_dir'], p)) 
            and not p.startswith('__')
            and not p.startswith('.')
            and p not in ['__pycache__', '.DS_Store']
        ]
        
        if not problems:
            print(f"No valid datasets found in: {config['data_dir']}")
            return
        
        for problem in problems:
            config['problem'] = problem
            print(f"\n{'='*70}")
            print(f"Processing: {problem}")
            print(f"{'='*70}\n")
            
            train_loader, val_loader, test_loader = load_single_airport_data(config)
            Data = (train_loader, val_loader, test_loader)
            best_aggr_metrics_test, all_metrics = unified_train(config, Data)
            
            print_str = '\nBest Model Test Summary: '
            for k, v in best_aggr_metrics_test.items():
                if isinstance(v, float):
                    print_str += '{}: {:.4f} | '.format(k, v)
                else:
                    print_str += '{}: {} | '.format(k, v)
            print(print_str)
            
            output_dir = config.get('output_dir', 'output')
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, problem + '_output.txt')
            with open(output_file, 'w') as file:
                file.write(f"Problem: {problem}\n")
                file.write(f"Training Mode: {config.get('Training_mode', 'Unified')}\n")
                file.write("-" * 50 + "\n")
                for k, v in all_metrics.items():
                    if isinstance(v, float):
                        file.write(f'{k}: {v:.6f}\n')
                    else:
                        file.write(f'{k}: {v}\n')
            
            print(f"Results saved to: {output_file}")


if __name__ == '__main__':
    main()
