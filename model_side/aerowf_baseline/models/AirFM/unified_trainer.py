"""
Unified Trainer for AeroWF Model.

Supports contrastive learning pre-training and masked reconstruction pre-training.

Model architecture:
    Input → Dual-stream Encoders (Temporal/Spectral) → Hierarchical Aggregators → 
    Representation → Pre-training Tasks

Exogenous variables (METAR data) support:
    - METAR as context, not involved in masking/reconstruction
    - Only masks/reconstructs original time series data
    - Integrated via additive fusion with main representation

Supported training modes:
    1. pretrain: Contrastive learning pre-training
    2. masked_recon: Masked reconstruction pre-training
    3. supervised: Supervised learning
    4. linear_probe: Linear probing (frozen encoder)
    5. finetune: Fine-tuning

Usage:
    trainer = UnifiedTrainer(model, config, device)
    trainer.train(train_loader, val_loader, num_epochs)

Exogenous variable batch format:
    batch = (X, labels, idx) or batch = (X, labels, idx, exo_cat, exo_cont)
    - exo_cat: Categorical exogenous variables dict
    - exo_cont: Continuous exogenous variables dict
"""

import os
import time
import math
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

try:
    from models.AirFM.soft_dtw_cuda import SoftDTW
    from models.AirFM.fft_filter import filter_frequencies
    HAS_DTW = True
except ImportError:
    HAS_DTW = False

logger = logging.getLogger('__main__')


class UnifiedTrainer:
    """
    Unified Trainer for AeroWF Model.
    
    Supports multiple training modes, unified management through configuration.
    """
    
    def __init__(self, model, config, device='cuda'):
        """
        Args:
            model: UnifiedSeries2Vec model
            config: Configuration dictionary
            device: Training device
        """
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        # Training mode
        self.training_mode = config.get('training_mode', 'pretrain')
        
        # Optimizer
        self.optimizer = self._create_optimizer()
        
        # Learning rate scheduler
        self.scheduler = self._create_scheduler()
        
        # DTW-related (for original pre-training and unified pre-training)
        if HAS_DTW and self.training_mode in ['pretrain', 'unified_pretrain']:
            self.sdtw = SoftDTW(use_cuda=True, gamma=0.1)
            self.filter_frequencies = filter_frequencies
        
        # Loss function
        self.criterion = self._create_criterion()
        
        # Logging
        self.save_dir = config.get('save_dir', './checkpoints')
        os.makedirs(self.save_dir, exist_ok=True)
        
        if SummaryWriter:
            self.writer = SummaryWriter(
                log_dir=os.path.join(self.save_dir, 'logs')
            )
        else:
            self.writer = None
        
        # Best model tracking
        self.best_metric = float('inf') if self.training_mode in ['pretrain', 'masked_recon', 'unified_pretrain'] else 0
        self.patience_counter = 0
        self.patience = config.get('patience', 20)
        # Minimum improvement threshold: only improvements exceeding this value are considered "best"
        # Avoids frequent saving due to floating point precision differences
        self.min_delta = config.get('min_delta', 1e-4)
        
        # 训练统计
        self.epoch_metrics = OrderedDict()
        
        # 梯度裁剪
        self.grad_clip = config.get('grad_clip', 1.0)
        
        logger.info(f"[UnifiedTrainer] Mode: {self.training_mode}")
        logger.info(f"[UnifiedTrainer] Device: {device}")
    
    def _create_optimizer(self):
        """创建优化器"""
        lr = self.config.get('lr', 1e-3)
        weight_decay = self.config.get('weight_decay', 1e-5)
        optimizer_name = self.config.get('optimizer', 'AdamW')
        
        if optimizer_name == 'AdamW':
            return torch.optim.AdamW(
                self.model.parameters(), lr=lr, weight_decay=weight_decay
            )
        elif optimizer_name == 'RAdam':
            return torch.optim.RAdam(
                self.model.parameters(), lr=lr, weight_decay=weight_decay
            )
        else:
            return torch.optim.Adam(
                self.model.parameters(), lr=lr, weight_decay=weight_decay
            )
    
    def _create_scheduler(self):
        """创建学习率调度器（带Warmup）"""
        epochs = self.config.get('epochs', 100)
        min_lr = self.config.get('min_lr', 1e-6)
        warmup_epochs = self.config.get('warmup_epochs', 3)  # 前3个epoch预热
        
        # 使用带Warmup的余弦退火
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                # 线性预热: 从 0.1x 增长到 1x
                return 0.1 + 0.9 * epoch / warmup_epochs
            else:
                # 余弦退火
                progress = (epoch - warmup_epochs) / (epochs - warmup_epochs)
                return max(min_lr / self.config.get('lr', 1e-3), 
                          0.5 * (1 + math.cos(math.pi * progress)))
        
        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
    def _create_criterion(self):
        """创建损失函数"""
        if self.training_mode == 'supervised':
            return nn.CrossEntropyLoss()
        elif self.training_mode == 'contrastive':
            return ContrastiveLoss(
                temperature=self.config.get('temperature', 0.07)
            )
        else:
            return None  # masked_recon和pretrain在模型内部计算损失
    
    # ===================== 训练方法 =====================
    
    def train_epoch(self, train_loader, epoch):
        """训练一个epoch"""
        self.model.train()
        
        epoch_loss = 0
        num_batches = 0
        loss_components = OrderedDict()
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}')
        
        for batch in pbar:
            loss, loss_dict = self._train_step(batch)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)
            self.optimizer.step()
            
            # 记录
            with torch.no_grad():
                epoch_loss += loss.item()
                num_batches += 1
                
                for key, value in loss_dict.items():
                    if key not in loss_components:
                        loss_components[key] = 0
                    loss_components[key] += value
                
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # 平均
        epoch_loss /= num_batches
        for key in loss_components:
            loss_components[key] /= num_batches
        
        # 记录到TensorBoard
        if self.writer:
            self.writer.add_scalar('Train/loss', epoch_loss, epoch)
            for key, value in loss_components.items():
                self.writer.add_scalar(f'Train/{key}', value, epoch)
            self.writer.add_scalar('Train/lr', self.optimizer.param_groups[0]['lr'], epoch)
        
        self.epoch_metrics['train_loss'] = epoch_loss
        self.epoch_metrics.update({f'train_{k}': v for k, v in loss_components.items()})
        
        return epoch_loss
    
    def _train_step(self, batch):
        """单步训练"""
        if self.training_mode == 'unified_pretrain':
            return self._unified_pretrain_step(batch)
        elif self.training_mode == 'pretrain':
            # 重定向到 unified_pretrain
            logger.warning("'pretrain' training mode 已废弃，使用 'unified_pretrain' 替代")
            return self._unified_pretrain_step(batch)
        elif self.training_mode == 'masked_recon':
            return self._masked_recon_step(batch)
        elif self.training_mode == 'contrastive':
            return self._contrastive_step(batch)
        elif self.training_mode in ['supervised', 'finetune']:
            return self._supervised_step(batch)
        elif self.training_mode == 'linear_probe':
            return self._linear_probe_step(batch)
        else:
            raise ValueError(f"Unknown training mode: {self.training_mode}")
    
    def _parse_batch(self, batch):
        """
        解析batch数据，支持带/不带外生变量的格式
        
        支持的格式:
        - 字典格式（DownstreamDataset 返回）:
            {'x': ..., 'runway_mask': ..., 'label_2': ..., 'label_3': ...}
        - 元组格式（旧 MultiAirportDataset 返回）:
            - (X, labels, idx): 无外生变量（单机场/旧格式）
            - (X, labels, idx, exo_cat, exo_cont): 有外生变量（单机场/旧格式）
            - (X, labels, mask): 多机场混合格式（无外生变量）
            - (X, labels, mask, exo_cat, exo_cont): 多机场混合格式（有外生变量）
        
        Returns:
            X: 原始时序数据
            labels: 标签
            idx: 索引（或None，如果是多机场格式）
            node_mask: 节点掩码（或None，如果是旧格式）
            exo_cat: 离散外生变量字典（或None）
            exo_cont: 连续外生变量字典（或None）
        """
        # 处理字典格式的 batch（DownstreamDataset 返回）
        if isinstance(batch, dict):
            X = batch['x']
            # 标签：优先使用 label_2（2分类），否则 label_21（原始21类）
            if 'label_2' in batch:
                labels = batch['label_2']
            elif 'label_21' in batch:
                labels = batch['label_21']
            else:
                labels = None
            
            # 跑道掩码
            node_mask = batch.get('runway_mask', None)
            idx = None
            
            # 外生变量（如果存在）
            exo_cat = batch.get('exo_categorical', None)
            exo_cont = batch.get('exo_continuous', None)
            
            # 将外生变量移到设备（如果存在且是字典）
            if exo_cat is not None and isinstance(exo_cat, dict):
                exo_cat = {k: v.to(self.device) for k, v in exo_cat.items()}
            if exo_cont is not None and isinstance(exo_cont, dict):
                exo_cont = {k: v.to(self.device) for k, v in exo_cont.items()}
            
            return X, labels, idx, node_mask, exo_cat, exo_cont
        
        # 处理元组/列表格式的 batch（旧 MultiAirportDataset 返回）
        if len(batch) == 3:
            # 可能是 (X, labels, idx) 或 (X, labels, mask) 多机场格式
            X, labels, third = batch[0], batch[1], batch[2]
            
            # 判断是索引还是掩码：掩码是布尔张量
            if isinstance(third, torch.Tensor) and third.dtype == torch.bool:
                # 多机场格式: (X, labels, mask)
                node_mask = third
                idx = None
            else:
                # 旧格式: (X, labels, idx)
                node_mask = None
                idx = third
            
            exo_cat, exo_cont = None, None
            
        elif len(batch) == 5:
            # 旧格式: (X, labels, idx, exo_cat, exo_cont)
            X, labels, idx, exo_cat, exo_cont = batch
            node_mask = None
            
            # 将外生变量移到设备
            if exo_cat is not None and isinstance(exo_cat, dict):
                exo_cat = {k: v.to(self.device) for k, v in exo_cat.items()}
            if exo_cont is not None and isinstance(exo_cont, dict):
                exo_cont = {k: v.to(self.device) for k, v in exo_cont.items()}
                
        elif len(batch) == 6:
            # 新格式: (X, labels, mask, exo_cat, exo_cont, extra)
            X, labels, node_mask, exo_cat, exo_cont, _ = batch
            idx = None
            
            # 将外生变量移到设备
            if exo_cat is not None and isinstance(exo_cat, dict):
                exo_cat = {k: v.to(self.device) for k, v in exo_cat.items()}
            if exo_cont is not None and isinstance(exo_cont, dict):
                exo_cont = {k: v.to(self.device) for k, v in exo_cont.items()}
                
        else:
            raise ValueError(f"Unexpected batch format with {len(batch)} elements")
        
        return X, labels, idx, node_mask, exo_cat, exo_cont
    
    def _masked_recon_step(self, batch):
        """
        掩码重建步骤（支持外生变量和多机场掩码）
        
        外生变量处理:
        - 外生变量作为可见上下文，不参与掩码
        - 只掩码原始时序数据 X
        
        多机场处理:
        - 通过 node_mask 标记有效/虚拟节点
        - 掩码只应用于时间维度，不影响节点维度
        """
        X, labels, _, node_mask, exo_cat, exo_cont = self._parse_batch(batch)
        X = X.to(self.device)
        
        if node_mask is not None:
            node_mask = node_mask.to(self.device)
        
        # 前向传播：外生变量作为上下文，只对X进行掩码/重建
        total_loss, loss_dict = self.model(
            X, mode='masked_recon',
            exo_categorical=exo_cat,
            exo_continuous=exo_cont,
            node_mask=node_mask
        )
        return total_loss, loss_dict
    
    def _contrastive_step(self, batch):
        """对比学习步骤（支持外生变量和多机场掩码）"""
        X, labels, _, node_mask, exo_cat, exo_cont = self._parse_batch(batch)
        X = X.to(self.device)

        if node_mask is not None:
            node_mask = node_mask.to(self.device)

        # 数据增强（只对原数据增强，外生变量保持不变）
        X_aug1 = self._augment(X)
        X_aug2 = self._augment(X)

        # 前向传播（外生变量作为上下文，带掩码）
        proj1, proj2 = self.model(
            X_aug1, mode='contrastive', x2=X_aug2,
            exo_categorical=exo_cat,
            exo_continuous=exo_cont,
            node_mask=node_mask
        )

        # 计算损失
        loss = self.criterion(proj1, proj2)

        loss_dict = {
            'contrastive': loss.item(),
            'has_exo': exo_cat is not None or exo_cont is not None
        }
        return loss, loss_dict

    def _unified_pretrain_step(self, batch):
        """
        统一预训练步骤：掩码重建 + Batch-Batch 物理对比学习 联合优化

        Loss = λ_recon * loss_recon + λ_contrast * (loss_T + loss_F)
        """
        X, labels, _, node_mask, exo_cat, exo_cont = self._parse_batch(batch)
        X = X.to(self.device)

        if node_mask is not None:
            node_mask = node_mask.to(self.device)

        # 前向传播：调用统一预训练方法（带掩码）
        total_loss, loss_dict = self.model(
            X, mode='unified_pretrain',
            exo_categorical=exo_cat,
            exo_continuous=exo_cont,
            sdtw=self.sdtw if HAS_DTW else None,
            node_mask=node_mask
        )

        return total_loss, loss_dict
    
    def _supervised_step(self, batch):
        """有监督训练步骤（支持外生变量和多机场掩码）"""
        X, labels, _, node_mask, exo_cat, exo_cont = self._parse_batch(batch)
        X = X.to(self.device)
        labels = labels.to(self.device)
        
        if node_mask is not None:
            node_mask = node_mask.to(self.device)
        
        # 前向传播（外生变量作为上下文，带掩码）
        logits = self.model(
            X, mode='classify',
            exo_categorical=exo_cat,
            exo_continuous=exo_cont,
            node_mask=node_mask
        )
        
        # 损失
        loss = self.criterion(logits, labels)
        
        # 准确率
        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            acc = (preds == labels).float().mean().item()
        
        loss_dict = {
            'ce_loss': loss.item(),
            'accuracy': acc,
            'has_exo': exo_cat is not None or exo_cont is not None
        }
        
        return loss, loss_dict
    
    def _linear_probe_step(self, batch):
        """线性探测步骤（编码器冻结）"""
        # 确保编码器被冻结
        self.model.freeze_encoder()
        return self._supervised_step(batch)
    
    # ===================== 验证方法 =====================
    
    def validate(self, val_loader, epoch):
        """验证（支持外生变量）"""
        self.model.eval()
        
        val_loss = 0
        num_batches = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                # 解析batch（支持带/不带外生变量和多机场掩码）
                X, labels, _, node_mask, exo_cat, exo_cont = self._parse_batch(batch)
                X = X.to(self.device)
                labels = labels.to(self.device)
                
                if node_mask is not None:
                    node_mask = node_mask.to(self.device)
                
                if self.training_mode in ['supervised', 'finetune', 'linear_probe']:
                    logits = self.model(
                        X, mode='classify',
                        exo_categorical=exo_cat,
                        exo_continuous=exo_cont,
                        node_mask=node_mask
                    )
                    loss = self.criterion(logits, labels)
                    
                    preds = logits.argmax(dim=-1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    
                elif self.training_mode == 'masked_recon':
                    loss, _ = self.model(
                        X, mode='masked_recon',
                        exo_categorical=exo_cat,
                        exo_continuous=exo_cont,
                        node_mask=node_mask
                    )
                    
                elif self.training_mode == 'pretrain':
                    dist_T, dist_F, _, _ = self.model(
                        X, mode='pretrain',
                        exo_categorical=exo_cat,
                        exo_continuous=exo_cont,
                        node_mask=node_mask
                    )
                    # 简化验证：只计算距离矩阵的统计信息
                    loss = dist_T.mean() + dist_F.mean()

                elif self.training_mode == 'unified_pretrain':
                    loss, loss_dict = self.model(
                        X, mode='unified_pretrain',
                        exo_categorical=exo_cat,
                        exo_continuous=exo_cont,
                        sdtw=self.sdtw if HAS_DTW else None,
                        filter_frequencies=self.filter_frequencies if HAS_DTW else None,
                        node_mask=node_mask
                    )
                    # 使用 total loss 作为验证损失
                    loss = loss_dict.get('total', loss)
                
                val_loss += loss.item() if hasattr(loss, 'item') else loss
                num_batches += 1
        
        val_loss /= num_batches
        
        # 计算指标
        metrics = {'val_loss': val_loss}
        
        if self.training_mode in ['supervised', 'finetune', 'linear_probe']:
            acc = accuracy_score(all_labels, all_preds)
            f1 = f1_score(all_labels, all_preds, average='macro')
            metrics['val_accuracy'] = acc
            metrics['val_f1'] = f1
        
        # 记录到TensorBoard
        if self.writer:
            for key, value in metrics.items():
                self.writer.add_scalar(f'Val/{key}', value, epoch)
        
        self.epoch_metrics.update(metrics)
        
        return metrics
    
    # ===================== 完整训练流程 =====================
    
    def train(self, train_loader, val_loader=None, test_loader=None, num_epochs=None):
        """
        完整训练流程
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            test_loader: 测试数据加载器
            num_epochs: 训练轮数
        
        Returns:
            best_metrics: 最佳模型指标
            all_metrics: 所有指标
        """
        num_epochs = num_epochs or self.config.get('epochs', 100)
        
        logger.info("=" * 70)
        logger.info(f"开始训练: {num_epochs} epochs, 模式: {self.training_mode}")
        logger.info("=" * 70)
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            # 训练
            train_loss = self.train_epoch(train_loader, epoch)
            
            # 验证
            val_metrics = {}
            if val_loader:
                val_metrics = self.validate(val_loader, epoch)
            
            # 学习率调度
            self.scheduler.step()
            
            # 打印
            lr = self.optimizer.param_groups[0]['lr']
            log_str = f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f}"
            if val_metrics:
                log_str += f" | Val Loss: {val_metrics.get('val_loss', 0):.4f}"
                if 'val_accuracy' in val_metrics:
                    log_str += f" | Val Acc: {val_metrics['val_accuracy']:.4f}"
            log_str += f" | LR: {lr:.6f}"
            logger.info(log_str)
            
            # 保存最佳模型
            is_best = self._check_best(val_metrics, epoch)
            if is_best:
                self.save_checkpoint(epoch, is_best=True)
                logger.info(f"✓ 保存最佳模型")
            
            # 早停
            if self.patience_counter >= self.patience:
                logger.info(f"早停: {self.patience} epochs无改善")
                break
            
            # 定期保存
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch)
        
        # 训练结束
        total_time = time.time() - start_time
        logger.info("=" * 70)
        logger.info(f"训练完成! 总时间: {total_time/60:.2f} 分钟")
        
        # 在测试集上评估
        if test_loader:
            logger.info("在测试集上评估最佳模型...")
            self.load_checkpoint('best_model.pth')
            test_metrics = self.evaluate(test_loader)
            logger.info(f"测试结果: {test_metrics}")
            self.epoch_metrics.update({f'test_{k}': v for k, v in test_metrics.items()})
        
        if self.writer:
            self.writer.close()
        
        return self.epoch_metrics
    
    def evaluate(self, data_loader):
        """评估模型（支持外生变量）"""
        self.model.eval()

        # 掩码重建任务：评估重建质量
        if self.training_mode == 'masked_recon':
            return self.evaluate_reconstruction(data_loader)

        # 统一预训练任务：使用线性探测评估
        if self.training_mode == 'unified_pretrain':
            return self.evaluate_linear_probe(data_loader)

        # 对比学习预训练（原始距离学习）：使用线性探测评估
        if self.training_mode == 'pretrain':
            return self.evaluate_linear_probe(data_loader)

        # 时频对比学习预训练：使用线性探测评估
        # 有监督任务：评估分类指标
        all_preds = []
        all_labels = []
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in data_loader:
                X, labels, _, node_mask, exo_cat, exo_cont = self._parse_batch(batch)
                X = X.to(self.device)
                labels = labels.to(self.device)
                
                if node_mask is not None:
                    node_mask = node_mask.to(self.device)
                
                logits = self.model(
                    X, mode='classify',
                    exo_categorical=exo_cat,
                    exo_continuous=exo_cont,
                    node_mask=node_mask
                )
                loss = self.criterion(logits, labels)
                preds = logits.argmax(dim=-1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                total_loss += loss.item()
                num_batches += 1
        
        metrics = {
            'loss': total_loss / num_batches,
            'accuracy': accuracy_score(all_labels, all_preds),
            'f1_macro': f1_score(all_labels, all_preds, average='macro')
        }
        
        # 混淆矩阵
        cm = confusion_matrix(all_labels, all_preds)
        logger.info(f"混淆矩阵:\n{cm}")
        
        return metrics
    
    def evaluate_linear_probe(self, test_loader, train_loader=None):
        """线性探测评估（用于预训练模型）"""
        if train_loader is None:
            logger.warning("线性探测需要训练数据，使用测试数据进行评估")
            train_loader = test_loader
        
        self.model.eval()
        
        # 提取特征
        train_features, train_labels = self._extract_features(train_loader)
        test_features, test_labels = self._extract_features(test_loader)
        
        # 训练线性分类器
        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(train_features, train_labels)
        
        # 预测
        preds = clf.predict(test_features)
        
        metrics = {
            'accuracy': accuracy_score(test_labels, preds),
            'f1_macro': f1_score(test_labels, preds, average='macro')
        }
        
        return metrics
    
    def evaluate_reconstruction(self, test_loader):
        """
        评估掩码重建质量（用于masked_recon任务，支持外生变量）
        
        简化版: FreTS输出已是时域，融合后只做时域重建
        
        外生变量处理:
        - 外生变量作为可见上下文，不参与掩码
        - 只评估原数据的重建质量
        
        数据流:
            x_masked → encoder_T → h_T ─┐
                                        ├─→ (+exo) → h_F + h_T → decoder → recon
            x_masked → encoder_F → h_F ─┘
        
        Args:
            test_loader: 测试数据加载器
        
        Returns:
            metrics: 重建指标字典
                - test_recon_loss: 重建损失
        """
        self.model.eval()
        
        total_loss = 0
        recon_loss_list = []
        num_batches = 0
        
        with torch.no_grad():
            for batch in test_loader:
                X, labels, _, node_mask, exo_cat, exo_cont = self._parse_batch(batch)
                X = X.to(self.device)
                
                if node_mask is not None:
                    node_mask = node_mask.to(self.device)
                
                # 获取重建结果（外生变量作为上下文，带掩码）
                # loss_dict: {'total': ..., 'recon': ..., 'has_exogenous': ...}
                loss, loss_dict = self.model(
                    X, mode='masked_recon',
                    exo_categorical=exo_cat,
                    exo_continuous=exo_cont,
                    node_mask=node_mask
                )
                total_loss += loss.item()
                
                # 收集重建损失
                if 'recon' in loss_dict:
                    recon_loss_list.append(loss_dict['recon'])
                
                num_batches += 1
        
        # 汇总指标
        metrics = {
            'test_recon_loss': total_loss / num_batches,
        }
        
        if recon_loss_list:
            metrics['recon_mse'] = np.mean(recon_loss_list)
        
        return metrics
    
    
    def _extract_features(self, data_loader):
        """提取特征（支持外生变量和多机场掩码）"""
        features = []
        labels = []

        with torch.no_grad():
            for batch in data_loader:
                X, y, _, node_mask, exo_cat, exo_cont = self._parse_batch(batch)
                X = X.to(self.device)
                
                if node_mask is not None:
                    node_mask = node_mask.to(self.device)

                # 时频对比学习模式：使用融合表示 Z
                feat = self.model.get_representation(
                    X,
                    exo_categorical=exo_cat,
                    exo_continuous=exo_cont,
                    node_mask=node_mask
                )
                features.append(feat.cpu().numpy())
                labels.extend(y.numpy())

        return np.vstack(features), np.array(labels)

    # ===================== 辅助方法 =====================

    @staticmethod
    def generate_list(num):
        """
        生成索引对列表（与Series2Vec完全一致）

        Args:
            num: 生成的数量

        Returns:
            索引对列表，例如：[(1,0), (2,0), (2,1), (3,0), (3,1), (3,2)]
        """
        result = []
        for i in range(1, num + 1):
            for j in range(0, i):
                result.append((i, j))
        return result

    def _normalize(self, x):
        """
        归一化到[0, 1]（与Series2Vec的Distance_normalizer一致）

        Args:
            x: 输入张量

        Returns:
            归一化后的张量
        """
        if len(x) == 1:
            return x / (x + 1e-8)
        else:
            min_val = torch.min(x)
            max_val = torch.max(x)
            # 归一化到[0, 1]
            return (x - min_val) / (max_val - min_val + 1e-8)
    
    def _compute_dtw_distance(self, X, index=None):
        """计算DTW距离（使用与Series2Vec一致的索引方式）"""
        batch_size = X.shape[0]
        if index is None:
            index = self.generate_list(batch_size - 1)

        # 使用索引提取样本对
        combination1 = X[[i[0] for i in index]].to('cuda')
        combination2 = X[[i[1] for i in index]].to('cuda')
        Dtw_Distance = self.sdtw(combination1, combination2)

        return Dtw_Distance
    
    def _compute_euclidean_distance(self, X, index=None):
        """计算欧氏距离（使用与Series2Vec一致的索引方式）"""
        batch_size = X.shape[0]
        if index is None:
            index = self.generate_list(batch_size - 1)

        # 使用索引提取样本对
        combination1 = X[[i[0] for i in index]].to('cuda')
        combination2 = X[[i[1] for i in index]].to('cuda')
        combination1_flat = combination1.view(combination1.size(0), -1)
        combination2_flat = combination2.view(combination2.size(0), -1)
        distances = torch.norm(combination1_flat - combination2_flat, dim=1)

        return distances
    
    def _augment(self, x):
        """
        增强版数据增强（用于对比学习和防过拟合）
        
        增强策略:
        1. 高斯噪声 (概率0.8)
        2. 时间偏移 (概率0.5)
        3. 时间缩放/扭曲 (概率0.3)
        4. 随机遮蔽 (概率0.3)
        5. 通道Dropout (概率0.2)
        """
        augmented = x.clone()
        
        # 1. 高斯噪声 (80%概率)
        if torch.rand(1).item() < 0.8:
            noise_scale = torch.rand(1).item() * 0.15 + 0.05  # 0.05~0.2
            noise = torch.randn_like(augmented) * noise_scale
            augmented = augmented + noise
        
        # 2. 随机时间偏移 (50%概率)
        if torch.rand(1).item() < 0.5:
            shift = torch.randint(-8, 9, (1,)).item()
            if shift != 0:
                augmented = torch.roll(augmented, shifts=shift, dims=-1)
        
        # 3. 时间缩放/扭曲 (30%概率) - 通过插值实现
        if torch.rand(1).item() < 0.3:
            scale = torch.rand(1).item() * 0.2 + 0.9  # 0.9~1.1
            # 简化实现：只做幅度缩放
            augmented = augmented * scale
        
        # 4. 随机时间段遮蔽 (30%概率)
        if torch.rand(1).item() < 0.3:
            T = augmented.shape[-1]
            mask_len = int(T * (torch.rand(1).item() * 0.1 + 0.05))  # 5%~15%
            start = torch.randint(0, max(1, T - mask_len), (1,)).item()
            augmented[..., start:start+mask_len] = 0
        
        # 5. 通道Dropout (20%概率)
        if torch.rand(1).item() < 0.2 and augmented.dim() >= 2:
            if augmented.dim() == 3:  # (B, C, T)
                C = augmented.shape[1]
                drop_idx = torch.randint(0, C, (1,)).item()
                augmented[:, drop_idx, :] = 0
            elif augmented.dim() == 4:  # (B, N, T, C)
                C = augmented.shape[-1]
                drop_idx = torch.randint(0, C, (1,)).item()
                augmented[..., drop_idx] = 0
        
        return augmented
    
    def _check_best(self, val_metrics, epoch):
        """
        检查是否为最佳模型
        
        使用 min_delta 阈值避免因微小波动频繁保存：
        - 对于 loss：只有 current < best - min_delta 才认为有改善
        - 对于 accuracy：只有 current > best + min_delta 才认为有改善
        """
        if not val_metrics:
            return False
        
        # 根据训练模式选择比较指标
        if self.training_mode in ['supervised', 'finetune', 'linear_probe']:
            current = val_metrics.get('val_accuracy', 0)
            # 准确率需要提升超过 min_delta 才算改善
            is_best = current > self.best_metric + self.min_delta
        else:
            current = val_metrics.get('val_loss', float('inf'))
            # 损失需要下降超过 min_delta 才算改善
            is_best = current < self.best_metric - self.min_delta
        
        if is_best:
            self.best_metric = current
            self.patience_counter = 0
        else:
            self.patience_counter += 1
        
        return is_best
    
    def save_checkpoint(self, epoch, is_best=False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_metric': self.best_metric,
            'config': self.config
        }
        
        if is_best:
            path = os.path.join(self.save_dir, 'best_model.pth')
        else:
            path = os.path.join(self.save_dir, f'checkpoint_epoch_{epoch+1}.pth')
        
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, filename):
        """加载检查点"""
        path = os.path.join(self.save_dir, filename)
        if not os.path.exists(path):
            logger.warning(f"检查点不存在: {path}")
            return
        
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_metric = checkpoint['best_metric']
        
        logger.info(f"✓ 已加载检查点: {path}")


class ContrastiveLoss(nn.Module):
    """对比学习损失 (InfoNCE)"""
    
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, z1, z2):
        """
        Args:
            z1, z2: (batch, dim) 归一化后的特征
        """
        batch_size = z1.size(0)
        
        # 拼接
        z = torch.cat([z1, z2], dim=0)  # (2N, D)
        
        # 相似度矩阵
        sim = torch.mm(z, z.t()) / self.temperature  # (2N, 2N)
        
        # 正样本mask
        pos_mask = torch.zeros((2 * batch_size, 2 * batch_size), dtype=torch.bool, device=z.device)
        pos_mask[range(batch_size), range(batch_size, 2 * batch_size)] = True
        pos_mask[range(batch_size, 2 * batch_size), range(batch_size)] = True
        
        # 对角线mask
        diag_mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
        
        # 计算损失
        pos_sim = sim[pos_mask].view(2 * batch_size, -1)
        neg_mask = ~diag_mask & ~pos_mask
        neg_sim = sim[neg_mask].view(2 * batch_size, -1)
        
        logits = torch.cat([pos_sim, neg_sim], dim=1)
        labels = torch.zeros(2 * batch_size, dtype=torch.long, device=z.device)
        
        loss = F.cross_entropy(logits, labels)
        return loss


# ===================== 测试代码 =====================
if __name__ == '__main__':
    from models.Series2Vec.unified_model import UnifiedSeries2Vec
    from torch.utils.data import DataLoader, TensorDataset
    
    print("=" * 70)
    print("测试 UnifiedTrainer")
    print("=" * 70)
    
    # 创建模拟数据
    X_train = torch.randn(100, 11, 96)
    y_train = torch.randint(0, 5, (100,))
    X_test = torch.randn(20, 11, 96)
    y_test = torch.randint(0, 5, (20,))
    
    train_dataset = TensorDataset(X_train, y_train, torch.arange(100))
    test_dataset = TensorDataset(X_test, y_test, torch.arange(20))
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16)
    
    # 配置
    config = {
        'Data_shape': (16, 11, 96),
        'emb_size': 32,
        'rep_size': 64,
        'num_heads': 4,
        'dim_ff': 128,
        'dropout': 0.1,
        'num_labels': 5,
        'use_hierarchical': False,
        'use_masked_recon': True,
        'training_mode': 'supervised',
        'epochs': 3,
        'lr': 1e-3,
        'save_dir': './test_checkpoints'
    }
    
    # 创建模型和训练器
    model = UnifiedSeries2Vec(config, num_classes=5)
    trainer = UnifiedTrainer(model, config, device='cpu')
    
    # 训练
    metrics = trainer.train(train_loader, test_loader, test_loader, num_epochs=3)
    
    print(f"\n最终指标: {metrics}")
    print("\n✅ 测试通过!")
    print("=" * 70)
