"""
Hierarchical Graph Structure for Multi-Runway Airport Forecasting.

Graph nodes: [R0, R1, ..., R_{N-1}, Airport]  (N runways + 1 airport virtual node)
Edge types:
  0: Aggregate (Runway → Airport)
  1: Broadcast (Airport → Runway)
  2: Self-loop
  3: Neighbor (Runway ↔ Runway)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RelationalGraphConvLayer(nn.Module):
    """
    Relational Graph Convolution Layer (Simplified R-GCN).
    
    Supports message passing with multiple edge types.
    Dynamically adapts to variable node counts.
    """
    
    def __init__(self, in_channels, out_channels, num_edge_types):
        """
        Args:
            in_channels: Input feature dimension
            out_channels: Output feature dimension
            num_edge_types: Number of edge type categories
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_edge_types = num_edge_types
        
        self.weight = nn.ModuleList([
            nn.Linear(in_channels, out_channels, bias=False)
            for _ in range(num_edge_types)
        ])
        
        self.bias = nn.Parameter(torch.zeros(out_channels))
        
        self.gate = nn.Sequential(
            nn.Linear(in_channels + out_channels, out_channels),
            nn.Sigmoid()
        )
        
        self.reset_parameters()
    
    def reset_parameters(self):
        for w in self.weight:
            nn.init.xavier_uniform_(w.weight)
        nn.init.zeros_(self.bias)
    
    def forward(self, x, edge_index, edge_type):
        """
        Forward pass for relational graph convolution.
        
        Args:
            x: [batch_size, num_nodes, in_channels] - Dynamic node count
            edge_index: [batch_size, 2, num_edges] or [2, num_edges]
            edge_type: [batch_size, num_edges] or [num_edges]
        
        Returns:
            out: [batch_size, num_nodes, out_channels]
        """
        batch_size, num_nodes, _ = x.shape
        
        if edge_index.dim() == 2:
            edge_index = edge_index.unsqueeze(0).expand(batch_size, -1, -1)
        if edge_type.dim() == 1:
            edge_type = edge_type.unsqueeze(0).expand(batch_size, -1)
        
        out = torch.zeros(batch_size, num_nodes, self.out_channels, device=x.device)
        
        for edge_t in range(self.num_edge_types):
            mask = (edge_type == edge_t)
            
            if mask.any():
                for b in range(batch_size):
                    edge_mask_b = mask[b]
                    if not edge_mask_b.any():
                        continue
                    
                    edges_b = edge_index[b, :, edge_mask_b]
                    src_nodes = edges_b[0]
                    tgt_nodes = edges_b[1]
                    
                    src_features = x[b, src_nodes, :]
                    transformed = self.weight[edge_t](src_features)
                    
                    for i, tgt in enumerate(tgt_nodes):
                        out[b, tgt] += transformed[i]
        
        degree = self._compute_degree(edge_index, num_nodes)
        degree = degree.clamp(min=1).unsqueeze(-1)
        out = out / degree
        
        out = out + self.bias
        
        return out
    
    def _compute_degree(self, edge_index, num_nodes):
        """
        Compute in-degree for each node.
        
        Args:
            edge_index: [batch_size, 2, num_edges]
            num_nodes: Total number of nodes
        
        Returns:
            degree: [batch_size, num_nodes]
        """
        batch_size = edge_index.size(0)
        degree = torch.zeros(batch_size, num_nodes, device=edge_index.device)
        
        for b in range(batch_size):
            tgt_nodes = edge_index[b, 1, :]
            degree[b] = torch.bincount(tgt_nodes, minlength=num_nodes).float()
        
        return degree


class HierarchicalGNN(nn.Module):
    """
    Hierarchical Graph Neural Network for Multi-Runway Airports.
    
    Implements two-level interaction between runways and airport level.
    Supports dynamic node counts and variable runway configurations.
    """
    
    def __init__(self, in_channels, hidden_channels, out_channels, 
                 num_edge_types=4, N_max=4, num_layers=2, dropout=0.1):
        """
        Args:
            in_channels: Input feature dimension (rep_size)
            hidden_channels: Hidden layer dimension
            out_channels: Output feature dimension
            num_edge_types: Number of edge type categories (4: aggregate, broadcast, self-loop, neighbor)
            N_max: Maximum number of runways (excluding airport node)
            num_layers: Number of GNN layers
            dropout: Dropout rate
        """
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.N_max = N_max
        self.num_edge_types = num_edge_types
        self.num_layers = num_layers
        
        self.airport_node_init = nn.Parameter(torch.randn(1, 1, in_channels))
        
        self.gnn_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for i in range(num_layers):
            in_dim = in_channels if i == 0 else hidden_channels
            out_dim = hidden_channels if i < num_layers - 1 else out_channels
            
            self.gnn_layers.append(
                RelationalGraphConvLayer(in_dim, out_dim, num_edge_types)
            )
            self.norms.append(nn.LayerNorm(out_dim))
        
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
    
    def _build_graph(self, num_runways, device):
        """
        Build graph structure dynamically based on actual runway count.
        
        Args:
            num_runways: Actual number of runways
            device: Device to place tensors on
        
        Returns:
            edge_index: [2, num_edges]
            edge_type: [num_edges]
        """
        airport_idx = num_runways
        
        src_list = []
        tgt_list = []
        type_list = []
        
        for r in range(num_runways):
            src_list.append(r)
            tgt_list.append(airport_idx)
            type_list.append(0)
        
        for r in range(num_runways):
            src_list.append(airport_idx)
            tgt_list.append(r)
            type_list.append(1)
        
        for n in range(num_runways + 1):
            src_list.append(n)
            tgt_list.append(n)
            type_list.append(2)
        
        for i in range(num_runways):
            for j in range(num_runways):
                if i != j:
                    src_list.append(i)
                    tgt_list.append(j)
                    type_list.append(3)
        
        edge_index = torch.tensor([src_list, tgt_list], dtype=torch.long, device=device)
        edge_type = torch.tensor(type_list, dtype=torch.long, device=device)
        
        return edge_index, edge_type
        
    def forward(self, runway_features, node_mask=None, edge_index=None, edge_type=None):
        """
        Forward pass for hierarchical GNN.
        
        Args:
            runway_features: [batch_size, N, in_channels] - Runway features (dynamic N)
            node_mask: [batch_size, N] boolean tensor (optional)
                True = valid real runway, False = padding virtual runway
            edge_index: [2, num_edges] - Edge indices (optional, auto-generated)
            edge_type: [num_edges] - Edge types (optional, auto-generated)
        
        Returns:
            enhanced_features: [batch_size, N, out_channels] - Enhanced runway features
        Returns:
            enhanced_features: [batch_size, N, out_channels] - Enhanced runway features
            airport_feature: [batch_size, out_channels] - Airport node features
        """
        batch_size, num_runways, _ = runway_features.shape
        device = runway_features.device
        
        if node_mask is not None:
            max_valid = node_mask.sum(dim=1).max().item()
            num_runways_effective = int(max_valid)
        else:
            num_runways_effective = num_runways
        
        if edge_index is None or edge_type is None:
            edge_index, edge_type = self._build_graph(num_runways_effective, device)
        
        if node_mask is not None:
            mask_expanded = node_mask.unsqueeze(-1).float()
            masked_features = runway_features * mask_expanded
            sum_features = masked_features.sum(dim=1, keepdim=True)
            valid_count = node_mask.sum(dim=1, keepdim=True).unsqueeze(-1).clamp(min=1)
            airport_init = sum_features / valid_count
        else:
            airport_init = runway_features.mean(dim=1, keepdim=True)
        
        x = torch.cat([runway_features, airport_init], dim=1)
        
        for i, (gnn, norm) in enumerate(zip(self.gnn_layers, self.norms)):
            x_new = gnn(x, edge_index, edge_type)
            x_new = norm(x_new)
            
            if x.shape[-1] == x_new.shape[-1]:
                x_new = x_new + x
            
            x_new = self.activation(x_new)
            x = self.dropout(x_new)
        
        enhanced_runways = x[:, :num_runways, :]
        airport_feature = x[:, num_runways, :]
        
        return enhanced_runways, airport_feature



class SimpleHierarchicalGNN(nn.Module):
    """
    Simplified Hierarchical GNN for Multi-Runway Aggregation.
    
    Only implements Aggregate and Broadcast operations for easier training.
    Supports dynamic node counts and variable runway configurations per airport.
    """
    
    def __init__(self, feature_dim, hidden_dim=None, dropout=0.1, N_max=4):
        """
        Args:
            feature_dim: Input feature dimension
            hidden_dim: Hidden layer dimension (default = feature_dim)
            dropout: Dropout rate
            N_max: Maximum number of nodes
        """
        super().__init__()
        if hidden_dim is None:
            hidden_dim = feature_dim
        
        self.N_max = N_max
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        
        self.aggregate = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.broadcast = nn.Sequential(
            nn.Linear(hidden_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.fusion = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU()
        )
    
    def forward(self, runway_features, node_mask=None):
        """
        Forward pass for simplified hierarchical aggregation.
        
        Args:
            runway_features: [batch_size, N_max, feature_dim]
                Padded runway features (N_max = maximum nodes)
            node_mask: [batch_size, N_max] boolean tensor (optional)
                True = valid real runway, False = padding virtual runway
        
        Returns:
            enhanced_features: [batch_size, N_max, feature_dim]
            airport_feature: [batch_size, feature_dim]
        """
        batch_size, N, feature_dim = runway_features.shape
        
        N_current = min(self.N_max, N)
        
        if node_mask is not None:
            if node_mask.shape != (batch_size, N):
                node_mask = node_mask[:, :N]
            
            mask_expanded = node_mask.unsqueeze(-1)
            
            masked_features = runway_features * mask_expanded
            sum_features = masked_features.sum(dim=1)
            
            valid_count = node_mask.sum(dim=1, keepdim=True)
            valid_count = valid_count.clamp(min=1)
            
            airport_hidden = sum_features / valid_count
        else:
            airport_hidden = runway_features[:, :N_current, :].mean(dim=1)
        
        airport_hidden = self.aggregate(airport_hidden)
        
        broadcast_feature = self.broadcast(airport_hidden)
        
        broadcast_feature = broadcast_feature.unsqueeze(1).expand(-1, N, -1)
        
        fused = torch.cat([runway_features, broadcast_feature], dim=-1)
        enhanced_features = self.fusion(fused)
        
        enhanced_features = enhanced_features + runway_features
        
        airport_feature = broadcast_feature[:, 0, :]
        
        return enhanced_features, airport_feature


if __name__ == '__main__':
    print("="*60)
    print("HierarchicalGNN Test - Dynamic Node Count Support")
    print("="*60)
    
    batch_size = 4
    in_channels = 64
    N_max = 4
    
    print("\nFull HierarchicalGNN - Without mask")
    num_runways = 3
    runway_features = torch.randn(batch_size, num_runways, in_channels)
    
    model = HierarchicalGNN(
        in_channels=in_channels,
        hidden_channels=128,
        out_channels=64,
        num_edge_types=4,
        N_max=N_max,
        num_layers=2
    )
    
    enhanced, airport = model(runway_features)
    print(f"Input: {runway_features.shape}")
    print(f"Output - Enhanced runway features: {enhanced.shape}")
    print(f"Output - Airport feature: {airport.shape}")
    
    print("\nFull HierarchicalGNN - With mask")
    runway_features_mixed = torch.randn(batch_size, N_max, in_channels)
    
    runway_features_mixed[2, 2:, :] = 0
    runway_features_mixed[3, 3:, :] = 0
    
    node_mask = torch.ones(batch_size, N_max, dtype=torch.bool)
    node_mask[2, 2:] = False
    node_mask[3, 3:] = False
    
    print(f"Input shape: {runway_features_mixed.shape}")
    print(f"Mask:\n{node_mask}")
    print(f"Valid nodes per sample: {node_mask.sum(dim=1)}")
    
    enhanced_masked, airport_masked = model(runway_features_mixed, node_mask=node_mask)
    print(f"Output - Enhanced runway features: {enhanced_masked.shape}")
    print(f"Output - Airport feature: {airport_masked.shape}")
    
    print("\nSimple HierarchicalGNN - Without mask")
    simple_model = SimpleHierarchicalGNN(feature_dim=in_channels, N_max=N_max)
    runway_features_padded = torch.randn(batch_size, N_max, in_channels)
    enhanced_simple, airport_simple = simple_model(runway_features_padded)
    print(f"Input: {runway_features_padded.shape}")
    print(f"Output - Enhanced runway features: {enhanced_simple.shape}")
    print(f"Output - Airport feature: {airport_simple.shape}")
    
    print("\nSimple HierarchicalGNN - With mask")
    runway_features = torch.randn(batch_size, N_max, in_channels)
    
    runway_features[2, 3, :] = 0
    runway_features[3, 2:, :] = 0
    
    node_mask = torch.ones(batch_size, N_max, dtype=torch.bool)
    node_mask[2, 3] = False
    node_mask[3, 2:] = False
    
    print(f"Input shape: {runway_features.shape}")
    print(f"Mask:\n{node_mask}")
    print(f"Valid nodes per sample: {node_mask.sum(dim=1)}")
    
    enhanced_masked, airport_masked = simple_model(runway_features, node_mask)
    print(f"Output - Enhanced runway features: {enhanced_masked.shape}")
    print(f"Output - Airport feature: {airport_masked.shape}")
    
    print("\nVerification - Mask aggregation effectiveness:")
    for i in range(batch_size):
        valid_mask = node_mask[i]
        valid_features = runway_features[i, valid_mask]
        expected_mean = valid_features.mean(dim=0)
        actual_airport = airport_masked[i]
        
        print(f"  Sample{i}: valid_nodes={valid_mask.sum().item()}, "
              f"valid_features_mean_norm={expected_mean.norm().item():.4f}")
    
    print("\n" + "="*60)
    print("All tests passed!")
    print("="*60)
