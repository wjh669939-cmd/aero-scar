"""
Physical Distance Computation Module for AeroWF.

Computes physical distance matrices for contrastive learning:
- Temporal domain: SoftDTW distance (captures sequence shape and dynamics)
- Spectral domain: FFT magnitude Euclidean distance (captures periodicity and energy)

Note: Uses EMA global statistics for normalization across batches.
"""

import torch
import torch.nn as nn


class PhysicsDistanceComputer(nn.Module):
    """
    Physics Distance Computor for AeroWF with EMA Normalization.
    
    Generates ground-truth distance matrices for contrastive learning:
    - Temporal: SoftDTW distance (captures sequence shape and dynamics)
    - Spectral: FFT magnitude Euclidean distance (captures periodicity and energy)
    
    Uses EMA (Exponential Moving Average) for global statistics normalization.
    """
    
    def __init__(self, sdtw_gamma=0.1, ema_momentum=0.99, warmup_batches=10):
        """
        Args:
            sdtw_gamma: SoftDTW gamma parameter
            ema_momentum: EMA momentum (default 0.99 for stability)
            warmup_batches: Number of batches for warmup phase
        """
        super().__init__()
        self.sdtw_gamma = sdtw_gamma
        self.ema_momentum = ema_momentum
        self.warmup_batches = warmup_batches
        
        # Buffer for EMA statistics (not involved in gradient, saved with model)
        # Temporal distance EMA statistics
        self.register_buffer('ema_dist_T_min', torch.tensor(0.0))
        self.register_buffer('ema_dist_T_max', torch.tensor(1.0))
        # Spectral distance EMA statistics
        self.register_buffer('ema_dist_F_min', torch.tensor(0.0))
        self.register_buffer('ema_dist_F_max', torch.tensor(1.0))
        # Number of batches processed (for warmup)
        self.register_buffer('num_batches_seen', torch.tensor(0))
        # Whether initialized
        self.register_buffer('initialized', torch.tensor(False))
    
    def _get_ema_momentum(self) -> float:
        """
        Get current EMA momentum.
        Uses smaller momentum during warmup phase for faster adaptation.
        """
        if self.num_batches_seen < self.warmup_batches:
            # Warmup phase: use linearly increasing momentum
            progress = self.num_batches_seen.float() / self.warmup_batches
            return 0.5 + (self.ema_momentum - 0.5) * progress
        return self.ema_momentum
    
    @torch.no_grad()
    def _update_ema_stats(self, batch_min: torch.Tensor, batch_max: torch.Tensor,
                          ema_min: torch.Tensor, ema_max: torch.Tensor) -> tuple:
        """
        Update EMA statistics.
        
        Args:
            batch_min: Current batch minimum
            batch_max: Current batch maximum
            ema_min: Current EMA minimum
            ema_max: Current EMA maximum
        
        Returns:
            Updated (ema_min, ema_max)
        """
        if not self.initialized:
            return batch_min.clone(), batch_max.clone()
        
        momentum = self._get_ema_momentum()
        new_min = momentum * ema_min + (1 - momentum) * batch_min
        new_max = momentum * ema_max + (1 - momentum) * batch_max
        
        return new_min, new_max
    
    def _normalize_with_ema(self, dist_matrix: torch.Tensor, 
                            ema_min: torch.Tensor, ema_max: torch.Tensor) -> torch.Tensor:
        """
        Normalize using EMA statistics.
        
        Args:
            dist_matrix: Original distance matrix
            ema_min: EMA minimum
            ema_max: EMA maximum
        
        Returns:
            Normalized distance matrix in [0, 1]
        """
        range_val = ema_max - ema_min
        if range_val < 1e-8:
            range_val = torch.tensor(1.0, device=dist_matrix.device)
        
        normalized = (dist_matrix - ema_min) / (range_val + 1e-8)
        
        return torch.clamp(normalized, 0.0, 1.0)
    
    def compute_temporal_distance(self, x: torch.Tensor, sdtw=None, node_mask=None) -> torch.Tensor:
        """
        Compute temporal physical distance matrix (B, B).
        
        Data flow: (B, N, T, C) → average real nodes → (B, T, C) → 
        reshape → SoftDTW/Euclidean → (B, B) → EMA normalization → [0, 1]
        
        Args:
            x: Input data (B, N, T, C)
            sdtw: SoftDTW computation function (optional)
            node_mask: Node mask (B, N), True=real node, False=virtual node
        """
        B, N, T, C = x.shape
        
        # Only use real nodes to compute distance
        if node_mask is not None:
            x_perm = x.permute(0, 2, 1, 3)
            x_mean_list = []
            for i in range(B):
                mask_i = node_mask[i]
                if mask_i.sum() > 0:
                    x_i = x_perm[i, :, mask_i, :]
                    x_mean_i = x_i.mean(dim=1)
                else:
                    x_mean_i = x_perm[i, :, 0, :]
                x_mean_list.append(x_mean_i)
            x_flat = torch.stack(x_mean_list, dim=0)
        else:
            x_perm = x.permute(0, 2, 1, 3)
            x_flat = x_perm.reshape(B, T, N * C)
        
        if sdtw is not None:
            # SoftDTW is symmetric, so only calculate the upper triangle.
            # Pairwise inputs are processed in chunks to avoid the original
            # B^2 Python calls and to keep GPU memory bounded.
            pair_chunk_size = max(
                1,
                int(
                    getattr(
                        self,
                        'softdtw_pair_chunk_size',
                        256,
                    )
                ),
            )

            pair_rows, pair_columns = torch.triu_indices(
                B,
                B,
                offset=0,
                device=x.device,
            )

            dist_matrix = torch.zeros(
                B,
                B,
                device=x.device,
                dtype=x.dtype,
            )

            pair_count = pair_rows.numel()

            for start in range(
                0,
                pair_count,
                pair_chunk_size,
            ):
                end = min(
                    start + pair_chunk_size,
                    pair_count,
                )

                rows = pair_rows[start:end]
                columns = pair_columns[start:end]

                pair_distances = sdtw(
                    x_flat[rows],
                    x_flat[columns],
                ).reshape(-1)

                dist_matrix[rows, columns] = pair_distances
                dist_matrix[columns, rows] = pair_distances
        else:
            # Alternative: use Euclidean distance as temporal distance
            x_flat_2d = x_flat.view(B, -1)
            dist_matrix = torch.cdist(x_flat_2d, x_flat_2d)
        
        # Fix 3: Use EMA normalization instead of Batch-wise MinMax
        batch_min = dist_matrix.min()
        batch_max = dist_matrix.max()
        
        # Update EMA statistics during training
        if self.training:
            new_min, new_max = self._update_ema_stats(
                batch_min, batch_max,
                self.ema_dist_T_min, self.ema_dist_T_max
            )
            self.ema_dist_T_min.copy_(new_min)
            self.ema_dist_T_max.copy_(new_max)
        
        # Normalize using EMA statistics
        dist_matrix = self._normalize_with_ema(
            dist_matrix, self.ema_dist_T_min, self.ema_dist_T_max
        )
        
        return dist_matrix
    
    def compute_frequency_distance(self, x: torch.Tensor, node_mask=None) -> torch.Tensor:
        """
        Compute frequency domain physical distance matrix (B, B)
        
        Fix 4: Support node_mask to filter virtual nodes
        
        Data flow:
        (B, N, T, C)
        -> Average real nodes for each sample -> (B, T, C)
        -> Permute -> (B, C, T)
        -> RFFT -> (B, C, T/2+1, complex)
        -> Abs -> (B, C, T/2+1)
        -> Reshape(B, -1) -> (B, C*T_freq)
        -> Euclidean -> (B, B)
        -> EMA normalization to [0, 1]
        
        Args:
            x: Input data (B, N, T, C)
            node_mask: Node mask (B, N), True=real node, False=virtual node
        """
        B, N, T, C = x.shape
        
        # 【修复问题4】：只用真实节点计算距离
        if node_mask is not None:
            # 对每个样本，只对真实节点取平均
            x_mean_list = []
            for i in range(B):
                mask_i = node_mask[i]  # (N,)
                if mask_i.sum() > 0:
                    x_i = x[i, mask_i, :, :]  # (num_real, T, C)
                    x_mean_i = x_i.mean(dim=0)  # (T, C) - 真实跑道平均
                else:
                    x_mean_i = x[i, 0, :, :]  # fallback: 使用第一个节点
                x_mean_list.append(x_mean_i)
            x_mean = torch.stack(x_mean_list, dim=0)  # (B, T, C)
            
            # Permute: (B, T, C) -> (B, C, T)
            x_perm = x_mean.permute(0, 2, 1)
        else:
            # 原行为：使用所有节点
            # Permute: (B, N, T, C) -> (B, N, C, T)
            x_perm = x.permute(0, 1, 3, 2)
        
        # RFFT: 实数FFT，只保留有效部分
        x_fft = torch.fft.rfft(x_perm, dim=-1)  # (B, C, T/2+1) 或 (B, N, C, T/2+1)
        
        # 取幅值
        x_mag = torch.abs(x_fft)
        
        # Flatten
        x_flat = x_mag.view(B, -1)
        
        # 欧氏距离: (B, B)
        dist_matrix = torch.cdist(x_flat, x_flat)
        
        # 【修复病灶三】：使用 EMA 归一化替代 Batch-wise MinMax
        batch_min = dist_matrix.min()
        batch_max = dist_matrix.max()
        
        # 训练模式下更新 EMA 统计量
        if self.training:
            new_min, new_max = self._update_ema_stats(
                batch_min, batch_max,
                self.ema_dist_F_min, self.ema_dist_F_max
            )
            self.ema_dist_F_min.copy_(new_min)
            self.ema_dist_F_max.copy_(new_max)
        
        # 使用 EMA 统计量归一化
        dist_matrix = self._normalize_with_ema(
            dist_matrix, self.ema_dist_F_min, self.ema_dist_F_max
        )
        
        return dist_matrix
    
    def forward(self, x: torch.Tensor, sdtw=None, node_mask=None) -> tuple:
        """
        同时计算时域和频域物理距离矩阵
        
        【修复问题4】：支持 node_mask 过滤虚拟节点
        
        Args:
            x: 输入数据 (B, N, T, C)
            sdtw: SoftDTW 计算函数（可选）
            node_mask: 节点掩码 (B, N)，True=真实节点，False=虚拟节点
        
        Returns:
            gt_dist_T: 时域 Ground Truth 距离矩阵 (B, B)
            gt_dist_F: 频域 Ground Truth 距离矩阵 (B, B)
        """
        gt_dist_T = self.compute_temporal_distance(x, sdtw=sdtw, node_mask=node_mask)
        gt_dist_F = self.compute_frequency_distance(x, node_mask=node_mask)
        
        # 更新 batch 计数和初始化标志
        if self.training:
            self.num_batches_seen.add_(1)
            if not self.initialized:
                self.initialized.fill_(True)
        
        return gt_dist_T, gt_dist_F
    
    def get_ema_stats(self) -> dict:
        """
        获取当前的 EMA 统计量（用于调试和监控）
        """
        return {
            'dist_T_min': self.ema_dist_T_min.item(),
            'dist_T_max': self.ema_dist_T_max.item(),
            'dist_F_min': self.ema_dist_F_min.item(),
            'dist_F_max': self.ema_dist_F_max.item(),
            'num_batches_seen': self.num_batches_seen.item(),
            'current_momentum': self._get_ema_momentum() if self.training else self.ema_momentum
        }


# 全局单例，确保 EMA 统计量在整个训练过程中持续更新
_global_physics_computer = None


def get_physics_computer(ema_momentum=0.99, warmup_batches=10, reset=False) -> PhysicsDistanceComputer:
    """
    获取全局物理距离计算器实例
    
    使用单例模式确保 EMA 统计量在整个训练过程中累积更新
    
    Args:
        ema_momentum: EMA 动量
        warmup_batches: 预热阶段的 batch 数
        reset: 是否重置（创建新实例）
    
    Returns:
        PhysicsDistanceComputer 实例
    """
    global _global_physics_computer
    
    if _global_physics_computer is None or reset:
        _global_physics_computer = PhysicsDistanceComputer(
            ema_momentum=ema_momentum,
            warmup_batches=warmup_batches
        )
    
    return _global_physics_computer


def compute_physical_distances(x: torch.Tensor, sdtw=None, node_mask=None) -> tuple:
    """
    便捷函数：计算物理距离矩阵
    
    【注意】：此函数创建临时实例，不会累积 EMA 统计量
    建议在模型中使用 get_physics_computer() 获取持久实例
    
    【修复问题4】：支持 node_mask 过滤虚拟节点
    
    Args:
        x: 输入数据 (B, N, T, C)
        sdtw: SoftDTW 计算函数（可选）
        node_mask: 节点掩码 (B, N)，True=真实节点，False=虚拟节点
    
    Returns:
        gt_dist_T: 时域距离矩阵 (B, B)
        gt_dist_F: 频域距离矩阵 (B, B)
    """
    # 使用全局单例以累积 EMA
    computer = get_physics_computer()
    # 确保与输入在同一设备上
    if next(computer.buffers()).device != x.device:
        computer = computer.to(x.device)
    return computer(x, sdtw=sdtw, node_mask=node_mask)
