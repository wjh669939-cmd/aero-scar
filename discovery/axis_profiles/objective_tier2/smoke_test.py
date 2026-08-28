"""CPU functional smoke gate for axis objective_tier2 (ENGINEERING-TIER, owner-maintained).
Usage: python smoke_test.py <edited_file.py>
NOTE Phase A: the runner still uses its embedded copy; keep both in sync until
Phase B switches the runner to load this file. Every change must be logged in CHANGELOG.md.
"""

import sys
sys.path.insert(0, "/root/autodl-tmp/aerowf_baseline/AeroWF")
import importlib.util
import torch
spec = importlib.util.spec_from_file_location("models.AirFM.unified_model_trial", sys.argv[1])
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)
torch.manual_seed(0)
# 缩小维度的真实构造：走与正式预训练相同的 unified_pretrain_forward 全路径
# （物理距离 GT + 混合掩码 + 编码 + 重建/对比双损失 + 反传）
config = {
    "Data_shape": (2, 4, 96, 11),
    "N_max": 4, "num_nodes": 3,
    "emb_size": 32, "rep_size": 64, "num_heads": 2, "dim_ff": 64,
    "dropout": 0.0,
    "encoder_num_heads": 2, "encoder_num_layers": 1, "encoder_dim_ff": 64,
    "frets_num_layers": 1,
    "use_simple_gnn": True, "gnn_hidden": 32, "gnn_layers": 1,
    "fusion_type": "residual_add", "use_frequency": True, "use_hierarchy": True,
    "use_masked_recon": True, "mask_ratio": 0.25,
    "random_mask_strategy": "random", "causal_mask_strategy": "last", "causal_prob": 0.5,
    "proj_hidden_dim": 32, "proj_output_dim": 16,
    "lambda_recon": 1.0, "lambda_contrast": 0.5,
    "ema_momentum": 0.99, "warmup_batches": 1,
    "sdtw_gamma": 0.1, "softdtw_pair_chunk_size": 8,
}
model = m.UnifiedSeries2Vec(config, num_classes=3)
model.train()
x = torch.rand(2, 4, 96, 11)
node_mask = torch.tensor([[True, True, True, False], [True, True, False, False]])
loss, loss_dict = model.unified_pretrain_forward(x, sdtw=None, node_mask=node_mask)
assert loss.dim() == 0 and torch.isfinite(loss), f"pretrain loss bad: {loss}"
loss.backward()
grads = [p.grad for p in model.parameters() if p.grad is not None]
assert grads, "no parameter received gradient"
assert all(torch.isfinite(g).all() for g in grads), "non-finite gradients"
assert isinstance(loss_dict, dict) and loss_dict, "loss_dict missing/empty"
print("SMOKE_OK")
