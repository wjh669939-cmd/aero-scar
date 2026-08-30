"""M-axis validators (ENGINEERING-TIER; DEC-001 activation prerequisites).

三道检查全部实装（2026-08-29）：
1. param_budget: |Δparams| / parent <= 5%（parent 基准 = 正式 pretrain 实测 3,930,853）
2. io_shapes + fwd_bwd_finite: 缩维配置实例化候选 unified_model，跑
   unified_pretrain_forward 前后向，校验输出结构 / 有限性 / 梯度
用法: python validators.py <candidate_unified_model.py>
"""

from __future__ import annotations

import sys

PARENT_PARAMETER_COUNT = 3930853  # 正式 parent pretrain 实测（seed 无关，结构决定）
PARAM_BUDGET_TOL = 0.05

_SMOKE_CONFIG = {
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


def param_budget(candidate_param_count: int,
                 parent_param_count: int = PARENT_PARAMETER_COUNT,
                 tol: float = PARAM_BUDGET_TOL) -> str | None:
    if parent_param_count <= 0:
        return "parent_param_count invalid"
    rel = abs(candidate_param_count - parent_param_count) / parent_param_count
    if rel > tol:
        return f"param budget exceeded: |delta|={rel:.4f} > {tol}"
    return None


def _load_candidate(path: str):
    import importlib.util
    sys.path.insert(0, "/root/autodl-tmp/aerowf_baseline/AeroWF")
    spec = importlib.util.spec_from_file_location("models.AirFM.unified_model_mtrial", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def io_shapes_and_fwd_bwd(candidate_path: str) -> str | None:
    """缩维前后向：返回错误文本或 None。注意参数预算须另用真实配置参数量判定。"""
    import torch
    module = _load_candidate(candidate_path)
    torch.manual_seed(0)
    model = module.UnifiedSeries2Vec(_SMOKE_CONFIG, num_classes=3)
    model.train()
    x = torch.rand(2, 4, 96, 11)
    node_mask = torch.tensor([[True, True, True, False], [True, True, False, False]])
    out = model.unified_pretrain_forward(x, sdtw=None, node_mask=node_mask)
    if not (isinstance(out, tuple) and len(out) == 2):
        return "unified_pretrain_forward must return (loss, loss_dict)"
    loss, loss_dict = out
    if loss.dim() != 0 or not torch.isfinite(loss):
        return f"loss invalid: {loss}"
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    if not grads or not all(torch.isfinite(g).all() for g in grads):
        return "gradients missing or non-finite"
    rep = model.encode(x, exo_categorical=None, exo_continuous=None, node_mask=node_mask)
    if rep is None:
        return "encode returned None"
    return None


def main() -> int:
    path = sys.argv[1]
    err = io_shapes_and_fwd_bwd(path)
    if err:
        print(f"M_VALIDATOR_FAIL: {err}")
        return 1
    print("M_VALIDATOR_OK (io/fwd/bwd; param budget checked separately on real config)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
