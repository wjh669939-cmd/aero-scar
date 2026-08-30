"""M 轴接口合同实测探针（只读，CPU，不动任何训练产物）。

1. 用正式 pretrain 真实配置实例化 UnifiedSeries2Vec，反推 sky 词表使参数量
   精确等于 parent 实测 3,930,853（配置保真校验）；
2. 对 M 轴编辑面模块挂 forward hook，跑一次真实形状前向，
   落盘各模块的输入/输出张量形状 → INTERFACE_CONTRACTS["model"] 的实测依据。
"""

from __future__ import annotations

import json
import sys

import torch

sys.path.insert(0, "/root/autodl-tmp/aerowf_baseline/AeroWF")

from models.AirFM.unified_model import UnifiedSeries2Vec

PARENT_PARAMS = 3_930_853
BATCH = 4  # 探针用小 batch，形状语义与正式训练一致

BASE_CONFIG = {
    "Data_shape": (BATCH, 4, 96, 11),
    "N_max": 4,
    "emb_size": 128, "rep_size": 256, "num_heads": 8, "dim_ff": 256, "dropout": 0.2,
    "encoder_num_heads": 4, "encoder_num_layers": 3, "encoder_dim_ff": 512,
    "frets_num_layers": 2,
    "use_simple_gnn": True, "gnn_hidden": 128, "gnn_layers": 2,
    "fusion_type": "residual_add", "use_frequency": True, "use_hierarchy": True,
    "use_masked_recon": True, "mask_ratio": 0.25,
    "random_mask_strategy": "random", "causal_mask_strategy": "last", "causal_prob": 0.5,
    "proj_hidden_dim": 256, "proj_output_dim": 128,
    "lambda_recon": 1.0, "lambda_contrast": 0.5,
    "ema_momentum": 0.99, "warmup_batches": 10,
    "sdtw_gamma": 0.1, "softdtw_pair_chunk_size": 256,
}


def make_config(sky_vocab: int) -> dict:
    cfg = dict(BASE_CONFIG)
    cfg["exo_config"] = {
        "categorical": {
            "significant_wx": {"vocab_size": 2},
            "sky_condition": {"vocab_size": sky_vocab},
            "has_gust": {"vocab_size": 2},
            "is_cavok": {"vocab_size": 2},
        },
        "continuous": ["visibility", "cloud_height", "gust_speed"],
    }
    return cfg


NUM_CLASSES = 21  # ckpt 对账实证：classifier=5397=21*(256+1)，正式 pretrain 用原始天气码 21 类
SKY_VOCAB = 1     # ckpt config 实证


def main() -> int:
    torch.manual_seed(0)
    model = UnifiedSeries2Vec(make_config(SKY_VOCAB), num_classes=NUM_CLASSES)
    n = sum(p.numel() for p in model.parameters())
    if n != PARENT_PARAMS:
        print(f"PARAM_MATCH_FAIL got={n} expect={PARENT_PARAMS}")
        return 1
    print(f"PARAM_MATCH num_classes={NUM_CLASSES} sky_vocab={SKY_VOCAB} params={n}")

    cfg = make_config(SKY_VOCAB)
    torch.manual_seed(0)
    model = UnifiedSeries2Vec(cfg, num_classes=NUM_CLASSES)
    model.train()

    shapes: dict[str, dict] = {}

    def fmt(t):
        if isinstance(t, torch.Tensor):
            return list(t.shape)
        if isinstance(t, (tuple, list)):
            return [fmt(x) for x in t]
        if isinstance(t, dict):
            return {k: fmt(v) for k, v in t.items()}
        return type(t).__name__

    def hook(name):
        def _h(mod, args, kwargs, out):
            shapes.setdefault(name, {
                "module": type(mod).__name__,
                "inputs": fmt(args),
                "kwargs": {k: fmt(v) for k, v in kwargs.items()},
                "output": fmt(out),
            })
        return _h

    surface_prefixes = ("fusion", "encoder", "frets", "exo", "transformer", "gnn", "pos")
    hooked = []
    for name, mod in model.named_modules():
        if not name:
            continue
        cls_module = type(mod).__module__
        if any(seg in cls_module for seg in ("fusion", "encoders")) and "." not in name:
            mod.register_forward_hook(hook(name), with_kwargs=True)
            hooked.append(f"{name} ({type(mod).__name__} from {cls_module})")
    # 顶层子模块兜底：编辑面模块可能被包了一层
    for name, mod in model.named_modules():
        cls_module = type(mod).__module__
        if any(seg in cls_module for seg in ("fusion", "encoders")) and name not in shapes:
            mod.register_forward_hook(hook(name), with_kwargs=True)
            if not any(h.startswith(name + " ") for h in hooked):
                hooked.append(f"{name} ({type(mod).__name__} from {cls_module})")

    x = torch.rand(BATCH, 4, 96, 11)
    node_mask = torch.ones(BATCH, 4, dtype=torch.bool)
    node_mask[:, 3] = False
    loss, loss_dict = model.unified_pretrain_forward(x, sdtw=None, node_mask=node_mask)
    loss.backward()
    print("FWD_BWD_OK loss=", float(loss))
    print("HOOKED:")
    for h in hooked:
        print(" ", h)
    out = {
        "sky_vocab": SKY_VOCAB, "num_classes": NUM_CLASSES, "param_count": PARENT_PARAMS,
        "probe_batch": BATCH, "loss_keys": sorted(loss_dict),
        "module_shapes": shapes,
    }
    path = "/root/autodl-tmp/clh_deploy/discovery/axis_profiles/model/measured_interface_v1.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("WROTE", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
