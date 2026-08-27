"""
AeroWF: Geometric-Aware Spectral-Temporal Dual-Stream Learning for Airport Weather Forecasting

Command-line argument configuration for AeroWF model training.

Model Architecture:
  - Temporal Branch: Transformer encoder capturing short-term dynamics
  - Spectral Branch: FreTS encoder (FFT→MLP→iFFT) capturing long-term patterns
  - Hierarchical Aggregator: GNN-based aggregation for variable-cardinality runway data
  
Pre-training Tasks:
  - pretrain: Contrastive learning via distance metrics (DTW + FFT)
  - masked_recon: Masked reconstruction with hybrid masking strategy
  - unified_pretrain: Joint optimization of reconstruction + contrastive learning (recommended)

Dual-Stream Fusion:
  - concat: Concatenate temporal and spectral representations
  - residual_add: Residual connection (recommended)
  - weighted_add: Weighted combination
  - gated: Gated fusion
"""

import argparse
import os
import json
from datetime import datetime
import torch
import logging

logging.basicConfig(format='%(asctime)s | %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)
parser = argparse.ArgumentParser(description='AeroWF Training')


def Initialization(args):
    """
    Initialize configuration from arguments.
    
    Args:
        args: Arguments object from argparse
    Returns:
        config: Configuration dictionary
    """
    config = args.args.__dict__

    initial_timestamp = datetime.now()
    output_dir = config['output_dir']
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    output_dir = os.path.join(output_dir, config['Training_mode'], config['dataset'],
                              initial_timestamp.strftime("%Y-%m-%d_%H-%M"))
    config['output_dir'] = output_dir
    if config.get('data_dir') is None:
        config['data_dir'] = os.getcwd() + '/Dataset/' + config['dataset']
    config['save_dir'] = os.path.join(output_dir, 'checkpoints')
    config['pred_dir'] = os.path.join(output_dir, 'predictions')
    config['tensorboard_dir'] = os.path.join(output_dir, 'tb_summaries')
    create_dirs([config['save_dir'], config['pred_dir'], config['tensorboard_dir']])

    with open(os.path.join(output_dir, 'configuration.json'), 'w') as fp:
        json.dump(config, fp, indent=4, sort_keys=True)
    logger.info("Stored configuration file in '{}'".format(output_dir))
    
    if config['seed'] is not None:
        torch.manual_seed(config['seed'])
    
    config['device'] = torch.device('cuda' if (torch.cuda.is_available() and config['gpu'] != '-1') else 'cpu')
    logger.info("Using device: {}".format(config['device']))
    
    return config


def create_dirs(dirs):
    """Create directories if they do not exist."""
    try:
        for dir_ in dirs:
            if not os.path.exists(dir_):
                os.makedirs(dir_)
        return 0
    except Exception as err:
        print("Creating directories error: {0}".format(err))
        exit(-1)


parser.add_argument('--data_dir', type=str, default=None,
                    help='Data directory path')
parser.add_argument('--dataset', default='airport', 
                    choices={'Benchmarks', 'UEA', 'UCR', 'airport'},
                    help='Dataset name (default: airport)')
parser.add_argument('--output_dir', default='Results',
                    help='Output directory')
parser.add_argument('--Norm', type=bool, default=False, 
                    help='Data normalization')
parser.add_argument('--val_ratio', type=float, default=0.2, 
                    help='Validation ratio')

parser.add_argument('--use_multi_airport', action='store_true',
                    help='Enable multi-airport joint training')
parser.add_argument('--N_max', type=int, default=4,
                    help='Maximum runway cardinality for multi-airport training (default: 4)')
parser.add_argument('--airport_list', type=str, default='ZBAA,ZSPD,ZBAD,ZSSS',
                    help='Airport list for training, comma-separated')
parser.add_argument('--mix_ratio', type=str, default='uniform',
                    choices={'uniform', 'balanced'},
                    help='Multi-airport mixing strategy: uniform or balanced')

parser.add_argument('--Training_mode', default='Unified', 
                    choices={'Pre_Training', 'Supervised', 'Unified'},
                    help='Training mode: Unified (recommended), Pre_Training, Supervised')
parser.add_argument('--task_type', default='masked_recon', 
                    choices={'pretrain', 'masked_recon', 'supervised', 'unified_pretrain'},
                    help='Task type: pretrain, masked_recon, supervised, unified_pretrain')
parser.add_argument('--Model_Type', default=['AeroWF'], 
                    choices={'AeroWF'},
                    help='Model type')

parser.add_argument('--emb_size', type=int, default=128, 
                    help='Embedding dimension')
parser.add_argument('--rep_size', type=int, default=256, 
                    help='Representation dimension')
parser.add_argument('--num_heads', type=int, default=8, 
                    help='Number of attention heads')
parser.add_argument('--dim_ff', type=int, default=256, 
                    help='Feed-forward dimension')
parser.add_argument('--dropout', type=float, default=0.2, 
                    help='Dropout ratio (default: 0.2-0.3)')

parser.add_argument('--use_transformer_encoder', type=bool, default=True, 
                    help='Use Transformer encoder (True) or CNN encoder (False)')
parser.add_argument('--encoder_num_heads', type=int, default=4, 
                    help='Number of attention heads in Transformer encoder')
parser.add_argument('--encoder_num_layers', type=int, default=3, 
                    help='Number of Transformer encoder layers (default: 4-12)')
parser.add_argument('--encoder_dim_ff', type=int, default=512, 
                    help='Transformer encoder FFN dimension')

parser.add_argument('--num_nodes', type=int, default=3, 
                    help='Number of nodes in hierarchical aggregator')
parser.add_argument('--use_simple_gnn', type=bool, default=True, 
                    help='Use simplified GNN (recommended)')
parser.add_argument('--gnn_hidden', type=int, default=128, 
                    help='GNN hidden dimension')
parser.add_argument('--gnn_layers', type=int, default=2, 
                    help='Number of GNN layers')

parser.add_argument('--fusion_type', default='residual_add',
                    choices={'concat', 'residual_add', 'weighted_add', 'gated'},
                    help='Dual-stream fusion method: concat, residual_add (recommended), weighted_add, gated')

parser.add_argument('--use_masked_recon', type=bool, default=True, 
                    help='Enable masked reconstruction decoder')
parser.add_argument('--mask_ratio', type=float, default=0.4, 
                    help='Masking ratio')
parser.add_argument('--random_mask_strategy', default='random', 
                    choices={'random', 'block', 'structured'}, 
                    help='Random masking strategy: random, block, structured')
parser.add_argument('--causal_mask_strategy', default='random_future',
                    choices={'last', 'random_future', 'half'},
                    help='Causal masking strategy: last, random_future, half')
parser.add_argument('--causal_prob', type=float, default=0.5,
                    help='Causal masking probability')

parser.add_argument('--lambda_recon', type=float, default=1.0,
                    help='Weight for reconstruction loss in unified pre-training')
parser.add_argument('--lambda_contrast', type=float, default=1.0,
                    help='Weight for contrastive loss in unified pre-training')

parser.add_argument('--epochs', type=int, default=100, 
                    help='Number of training epochs')
parser.add_argument('--batch_size', type=int, default=96, 
                    help='Batch size')
parser.add_argument('--lr', type=float, default=3e-4, 
                    help='Learning rate (default: 1e-4 to 5e-4)')
parser.add_argument('--weight_decay', type=float, default=1e-4, 
                    help='Weight decay (L2 regularization)')
parser.add_argument('--patience', type=int, default=10, 
                    help='Early stopping patience (default: 5-10)')
parser.add_argument('--min_delta', type=float, default=1e-4,
                    help='Minimum improvement threshold for saving best model')
parser.add_argument('--warmup_epochs', type=int, default=5,
                    help='Number of learning rate warmup epochs')
parser.add_argument('--grad_clip', type=float, default=3.0,
                    help='Gradient clipping threshold')

parser.add_argument('--do_finetune', type=bool, default=True, 
                    help='Whether to fine-tune after pre-training')
parser.add_argument('--finetune_epochs', type=int, default=50, 
                    help='Number of fine-tuning epochs')
parser.add_argument('--finetune_lr', type=float, default=1e-4, 
                    help='Fine-tuning learning rate')

parser.add_argument('--gpu', type=int, default=0, 
                    help='GPU index (-1 for CPU)')
parser.add_argument('--seed', default=1234, type=int, 
                    help='Random seed')
parser.add_argument('--print_interval', type=int, default=10, 
                    help='Print interval')

args = parser.parse_args()
