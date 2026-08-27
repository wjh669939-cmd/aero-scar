#!/usr/bin/env python3
"""Re-export full-coverage val predictions from an existing forecast run checkpoint.

Does not retrain. Writes validation_predictions.npz next to --output-dir
(default: <run>/validation_predictions_full.npz) so the original file is kept.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader

SRC = Path("/root/autodl-tmp/aerowf_downstream_v2/src")
sys.path.insert(0, str(SRC))

from aerowf_forecast_train_v2 import (  # noqa: E402
    AIRPORTS,
    AeroWFForecastModel,
    AirportForecastDataset,
    CONTRACT_VAL_ROWS,
    assert_full_val_export_coverage,
    collect_full_val_predictions,
    sha256_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Default: <run-dir>/validation_predictions_full.npz",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(
            "/root/autodl-tmp/aerowf_delivery/v1/extracted/"
            "AeroWF_v1_MODEL_TRAINING/release_v1/trainval"
        ),
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path("/root/autodl-tmp/aerowf_baseline/AeroWF"),
    )
    parser.add_argument(
        "--pretrain-checkpoint",
        type=Path,
        default=None,
        help="Architecture source; default: run config.json checkpoint or seed43 pretrain.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    ckpt_path = run_dir / "checkpoints" / "best_model.pth"
    if not ckpt_path.is_file():
        raise FileNotFoundError(ckpt_path)
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    pretrain_path = args.pretrain_checkpoint
    if pretrain_path is None:
        stored = config.get("checkpoint")
        pretrain_path = Path(stored) if stored else Path(
            "/root/autodl-tmp/aerowf_baseline/AeroWF/results/"
            "aerowf_unified_pretrain_full_formal_seed42_v1/checkpoints/best_model.pth"
        )
    if not pretrain_path.is_file():
        # scratch runs still store a default pretrain path used only for architecture
        fallback = Path(
            "/root/autodl-tmp/aerowf_baseline/AeroWF/results/"
            "aerowf_unified_pretrain_full_formal_seed42_v1/checkpoints/best_model.pth"
        )
        if fallback.is_file():
            pretrain_path = fallback
        else:
            raise FileNotFoundError(pretrain_path)

    out_path = (
        args.output.resolve()
        if args.output is not None
        else run_dir / "validation_predictions_full.npz"
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    sys.path.insert(0, str(args.repo_root))
    os.chdir(args.repo_root)
    from models.AirFM.unified_model import UnifiedSeries2Vec

    device = torch.device("cuda")
    export_parts = [
        AirportForecastDataset(
            args.data_root / "val", airport, require_complete_horizons=False
        )
        for airport in AIRPORTS
    ]
    loader = DataLoader(
        ConcatDataset(export_parts),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    pretrained_checkpoint = torch.load(
        pretrain_path, map_location="cpu", weights_only=False
    )
    core = UnifiedSeries2Vec(dict(pretrained_checkpoint["config"]), num_classes=21)
    model = AeroWFForecastModel(core).to(device)
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state_dict"], strict=True)

    arrays = collect_full_val_predictions(model, loader, device)
    assert_full_val_export_coverage(arrays)
    np.savez_compressed(out_path, **arrays)
    print(
        json.dumps(
            {
                "output": str(out_path),
                "rows": int(arrays["source_index"].shape[0]),
                "expected_rows": CONTRACT_VAL_ROWS,
                "sha256": sha256_file(out_path),
                "source_checkpoint": str(ckpt_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
