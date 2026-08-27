#!/usr/bin/env python3
"""AeroWF V2 paper-aligned runway wind forecasting baseline.

The script supports both train-from-scratch and pretrained initialization under
the same data construction, model head, optimizer, loss and evaluation code.
Only AeroWF v1 model-side train/validation data are accessed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import shutil
import subprocess
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from trial_features import build_forecast_inputs
from trial_objective import forecast_loss


AIRPORTS = ("ZBAA", "ZSPD", "ZSSS")
AIRPORT_TO_ID = {name: index for index, name in enumerate(AIRPORTS)}
HORIZONS = OrderedDict((("T+1", 15), ("T+4", 60), ("T+8", 120)))
COMPONENTS = OrderedDict((("wind_x", 1), ("wind_y", 2)))
TARGET_INTERNAL_INDEX = 95
NS_PER_MINUTE = 60 * 1_000_000_000
# Evaluator manifest: 3961 val rows × 3 airports. Train/val loss still drops
# rows that lack a complete T+1/T+4/T+8 partner; prediction export does not.
CONTRACT_VAL_ROWS_PER_AIRPORT = 3961
CONTRACT_VAL_ROWS = CONTRACT_VAL_ROWS_PER_AIRPORT * len(AIRPORTS)
WIND_DENOMINATORS = {
    "wind_x": 69.99781799316406,
    "wind_y": 60.59617042541504,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--initialization", choices=("scratch", "pretrained"), required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--formal", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument("--scratch-lr", type=float, default=1e-4)
    parser.add_argument("--encoder-lr", type=float, default=1e-5)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=3.0)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path("/root/autodl-tmp/aerowf_baseline/AeroWF"),
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(
            "/root/autodl-tmp/aerowf_delivery/v1/extracted/"
            "AeroWF_v1_MODEL_TRAINING/release_v1/trainval"
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "/root/autodl-tmp/aerowf_baseline/AeroWF/results/"
            "aerowf_unified_pretrain_full_formal_seed42_v1/"
            "checkpoints/best_model.pth"
        ),
    )
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path(
            "/root/autodl-tmp/aerowf_downstream_v2/contracts/"
            "DOWNSTREAM_TASK_CONTRACT_v2.json"
        ),
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while block := file.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def git_output(repo_root: Path, *arguments: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *arguments], cwd=repo_root, text=True, stderr=subprocess.STDOUT
        ).strip()
    except Exception as error:  # provenance must not make training fail
        return f"UNAVAILABLE: {type(error).__name__}: {error}"


class AirportForecastDataset(Dataset):
    """Timestamp-aligned multi-horizon samples for one airport and split.

    Training / D-side val metrics require a complete partner at every horizon.
    Prediction export sets require_complete_horizons=False so every source row
    with an input window is scored by the model (tail rows still have T+1/T+4
    partners on the evaluator).
    """

    def __init__(
        self,
        split_root: Path,
        airport: str,
        *,
        require_complete_horizons: bool = True,
    ):
        self.airport = airport
        self.airport_id = AIRPORT_TO_ID[airport]
        self.root = split_root / airport
        self.require_complete_horizons = require_complete_horizons

        self.runway = np.load(self.root / "runway.npy", mmap_mode="r")
        self.runway_mask = np.load(self.root / "runway_mask.npy", mmap_mode="r")
        self.timestamps = np.load(self.root / "timestamps.npy", mmap_mode="r")
        self.source_index = np.load(self.root / "source_index.npy", mmap_mode="r")
        self.weather_code = np.load(
            self.root / "exo_cat_weather_code_id.npy", mmap_mode="r"
        )
        self.sky_condition = np.load(
            self.root / "exo_cat_sky_condition.npy", mmap_mode="r"
        )
        self.has_gust = np.load(self.root / "exo_cat_has_gust.npy", mmap_mode="r")
        self.is_cavok = np.load(self.root / "exo_cat_is_cavok.npy", mmap_mode="r")
        self.exo_continuous = np.load(self.root / "exo_continuous.npy", mmap_mode="r")

        timestamp_ns = np.asarray(self.timestamps).astype("datetime64[ns]").astype(np.int64)
        if len(np.unique(timestamp_ns)) != len(timestamp_ns):
            raise RuntimeError(f"{self.root}: duplicate timestamps")
        lookup = {int(timestamp): index for index, timestamp in enumerate(timestamp_ns)}

        anchors: list[int] = []
        partners: list[list[int]] = []
        for anchor, timestamp in enumerate(timestamp_ns):
            partner_row = [
                lookup.get(int(timestamp + minutes * NS_PER_MINUTE), -1)
                for minutes in HORIZONS.values()
            ]
            if require_complete_horizons and not all(index >= 0 for index in partner_row):
                continue
            anchors.append(anchor)
            partners.append(partner_row)

        self.anchors = np.asarray(anchors, dtype=np.int64)
        self.partners = np.asarray(partners, dtype=np.int64)
        if self.partners.shape != (len(self.anchors), len(HORIZONS)):
            raise RuntimeError(f"{self.root}: invalid partner table {self.partners.shape}")

        for horizon_index, minutes in enumerate(HORIZONS.values()):
            valid = self.partners[:, horizon_index] >= 0
            if require_complete_horizons and not np.all(valid):
                raise RuntimeError(f"{self.root}: incomplete horizon partners")
            if not np.any(valid):
                continue
            actual = (
                timestamp_ns[self.partners[valid, horizon_index]]
                - timestamp_ns[self.anchors[valid]]
            ) // NS_PER_MINUTE
            if not np.all(actual == minutes):
                raise RuntimeError(f"{self.root}: {minutes}-minute alignment failed")

        anchor_masks = np.asarray(self.runway_mask[self.anchors], dtype=bool)
        for horizon_index in range(len(HORIZONS)):
            valid = self.partners[:, horizon_index] >= 0
            if not np.any(valid):
                continue
            partner_masks = np.asarray(
                self.runway_mask[self.partners[valid, horizon_index]], dtype=bool
            )
            if not np.array_equal(anchor_masks[valid], partner_masks):
                raise RuntimeError(f"{self.root}: runway masks differ across targets")

    def __len__(self) -> int:
        return len(self.anchors)

    def __getitem__(self, item: int) -> dict[str, Any]:
        anchor = int(self.anchors[item])
        partner = self.partners[item]
        target_slices = []
        for index in partner:
            if int(index) < 0:
                n_runways = int(self.runway.shape[1])
                target_slices.append(
                    np.full((n_runways, len(COMPONENTS)), np.nan, dtype=np.float32)
                )
            else:
                target_slices.append(
                    np.asarray(
                        self.runway[int(index), :, TARGET_INTERNAL_INDEX, 1:3],
                        dtype=np.float32,
                    )
                )
        target = np.stack(target_slices, axis=1)
        sample = build_forecast_inputs(
            self.runway[anchor],
            self.runway_mask[anchor],
            {
                "weather_code": int(self.weather_code[anchor]),
                "sky_condition": int(self.sky_condition[anchor]),
                "has_gust": int(self.has_gust[anchor]),
                "is_cavok": int(self.is_cavok[anchor]),
            },
            self.exo_continuous[anchor],
        )
        sample["target"] = torch.from_numpy(target)
        sample["airport_id"] = torch.tensor(self.airport_id, dtype=torch.long)
        sample["anchor_timestamp_ns"] = torch.tensor(
            int(np.asarray(self.timestamps[anchor]).astype("datetime64[ns]").astype(np.int64)),
            dtype=torch.long,
        )
        sample["source_index"] = torch.tensor(int(self.source_index[anchor]), dtype=torch.long)
        return sample


class AeroWFForecastModel(nn.Module):
    def __init__(self, core: nn.Module):
        super().__init__()
        self.core = core
        self.forecast_head = nn.Sequential(
            nn.LayerNorm(self.core.fusion.output_dim),
            nn.Linear(self.core.fusion.output_dim, len(HORIZONS) * len(COMPONENTS)),
        )

    def forward(
        self,
        x: torch.Tensor,
        node_mask: torch.Tensor,
        exo_categorical: dict[str, torch.Tensor],
        exo_continuous: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        _, _, intermediate = self.core.encode(
            x,
            return_intermediate=True,
            exo_categorical=exo_categorical,
            exo_continuous=exo_continuous,
            node_mask=node_mask,
        )
        z_t = intermediate["z_T_after_hier"]
        z_f = intermediate["z_F_after_hier"]
        batch_size, num_runways, _ = z_t.shape
        fused = self.core.fusion(
            z_t.reshape(batch_size * num_runways, -1),
            z_f.reshape(batch_size * num_runways, -1),
        )
        return self.forecast_head(fused).reshape(
            batch_size, num_runways, len(HORIZONS), len(COMPONENTS)
        )


def move_nested(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: move_nested(item, device) for key, item in value.items()}
    return value


def fresh_accumulator() -> dict[str, dict[str, dict[str, float | int]]]:
    return {
        horizon: {
            component: {"abs_norm": 0.0, "sq_norm": 0.0, "count": 0}
            for component in COMPONENTS
        }
        for horizon in HORIZONS
    }


def metrics_from_accumulator(
    accumulator: dict[str, dict[str, dict[str, float | int]]]
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    cell_mae_norm: list[float] = []
    cell_rmse_norm: list[float] = []
    cell_mae_physical: list[float] = []
    cell_rmse_physical: list[float] = []

    for horizon in HORIZONS:
        output[horizon] = {"components": {}}
        pooled_abs_norm = pooled_sq_norm = 0.0
        pooled_abs_physical = pooled_sq_physical = 0.0
        pooled_count = 0

        for component in COMPONENTS:
            row = accumulator[horizon][component]
            count = int(row["count"])
            denominator = WIND_DENOMINATORS[component]
            mae_norm = float(row["abs_norm"]) / count
            rmse_norm = math.sqrt(float(row["sq_norm"]) / count)
            mae_physical = mae_norm * denominator
            rmse_physical = rmse_norm * denominator
            output[horizon]["components"][component] = {
                "mae_norm": mae_norm,
                "rmse_norm": rmse_norm,
                "mae_mps": mae_physical,
                "rmse_mps": rmse_physical,
                "count": count,
            }
            cell_mae_norm.append(mae_norm)
            cell_rmse_norm.append(rmse_norm)
            cell_mae_physical.append(mae_physical)
            cell_rmse_physical.append(rmse_physical)
            pooled_abs_norm += float(row["abs_norm"])
            pooled_sq_norm += float(row["sq_norm"])
            pooled_abs_physical += float(row["abs_norm"]) * denominator
            pooled_sq_physical += float(row["sq_norm"]) * denominator**2
            pooled_count += count

        output[horizon]["paper_table_aggregate"] = {
            "mae_norm": pooled_abs_norm / pooled_count,
            "rmse_norm": math.sqrt(pooled_sq_norm / pooled_count),
            "mae_mps": pooled_abs_physical / pooled_count,
            "rmse_mps": math.sqrt(pooled_sq_physical / pooled_count),
            "count": pooled_count,
        }

    output["summary"] = {
        "MAE_macro_norm": float(np.mean(cell_mae_norm)),
        "RMSE_macro_norm": float(np.mean(cell_rmse_norm)),
        "MAE_macro_mps": float(np.mean(cell_mae_physical)),
        "RMSE_macro_mps": float(np.mean(cell_rmse_physical)),
    }
    return output


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float,
) -> float:
    model.train()
    squared_sum = 0.0
    element_count = 0

    for batch_index, batch in enumerate(loader, start=1):
        x = batch["x"].to(device, non_blocking=True)
        node_mask = batch["node_mask"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        exo_categorical = move_nested(batch["exo_categorical"], device)
        exo_continuous = move_nested(batch["exo_continuous"], device)

        optimizer.zero_grad(set_to_none=True)
        prediction = model(x, node_mask, exo_categorical, exo_continuous)
        loss = forecast_loss(prediction, target, node_mask)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite train loss at batch {batch_index}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        count = int(node_mask.sum().item()) * len(HORIZONS) * len(COMPONENTS)
        squared_sum += float(loss.item()) * count
        element_count += count

    return squared_sum / element_count


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    collect_predictions: bool = False,
) -> tuple[float, dict[str, Any], dict[str, np.ndarray] | None]:
    model.eval()
    accumulator = fresh_accumulator()
    total_squared = 0.0
    total_count = 0
    collected: dict[str, list[np.ndarray]] | None = None
    if collect_predictions:
        collected = {
            "prediction": [],
            "target": [],
            "node_mask": [],
            "airport_id": [],
            "anchor_timestamp_ns": [],
            "source_index": [],
        }

    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        node_mask = batch["node_mask"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        exo_categorical = move_nested(batch["exo_categorical"], device)
        exo_continuous = move_nested(batch["exo_continuous"], device)
        prediction = model(x, node_mask, exo_categorical, exo_continuous)
        if not torch.isfinite(prediction).all():
            raise RuntimeError("non-finite validation prediction")

        mask = node_mask[:, :, None, None].expand_as(prediction)
        error = prediction - target
        total_squared += float(torch.square(error)[mask].sum().item())
        total_count += int(mask.sum().item())

        for horizon_index, horizon in enumerate(HORIZONS):
            for component_index, component in enumerate(COMPONENTS):
                component_error = error[:, :, horizon_index, component_index]
                selected = component_error[node_mask]
                row = accumulator[horizon][component]
                row["abs_norm"] += float(torch.abs(selected).sum().item())
                row["sq_norm"] += float(torch.square(selected).sum().item())
                row["count"] += int(selected.numel())

        if collected is not None:
            collected["prediction"].append(prediction.cpu().numpy())
            collected["target"].append(target.cpu().numpy())
            collected["node_mask"].append(node_mask.cpu().numpy())
            for key in ("airport_id", "anchor_timestamp_ns", "source_index"):
                collected[key].append(batch[key].cpu().numpy())

    arrays = None
    if collected is not None:
        arrays = {key: np.concatenate(values) for key, values in collected.items()}
    return total_squared / total_count, metrics_from_accumulator(accumulator), arrays


@torch.no_grad()
def collect_full_val_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, np.ndarray]:
    """Inference-only export. Does not require complete horizon targets."""
    model.eval()
    collected: dict[str, list[np.ndarray]] = {
        "prediction": [],
        "target": [],
        "node_mask": [],
        "airport_id": [],
        "anchor_timestamp_ns": [],
        "source_index": [],
    }
    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        node_mask = batch["node_mask"].to(device, non_blocking=True)
        exo_categorical = move_nested(batch["exo_categorical"], device)
        exo_continuous = move_nested(batch["exo_continuous"], device)
        prediction = model(x, node_mask, exo_categorical, exo_continuous)
        if not torch.isfinite(prediction).all():
            raise RuntimeError("non-finite full-coverage validation prediction")
        collected["prediction"].append(prediction.cpu().numpy())
        collected["target"].append(batch["target"].cpu().numpy())
        collected["node_mask"].append(node_mask.cpu().numpy())
        for key in ("airport_id", "anchor_timestamp_ns", "source_index"):
            collected[key].append(batch[key].cpu().numpy())
    return {key: np.concatenate(values) for key, values in collected.items()}


def assert_full_val_export_coverage(arrays: dict[str, np.ndarray]) -> None:
    rows = int(arrays["source_index"].shape[0])
    if rows != CONTRACT_VAL_ROWS:
        raise RuntimeError(
            f"forecast export rows {rows} != {CONTRACT_VAL_ROWS} "
            "(evaluator requires full val coverage)"
        )
    if bool(np.isnan(arrays["prediction"]).any()) or bool(
        np.isinf(arrays["prediction"]).any()
    ):
        raise RuntimeError("full-coverage export contains NaN/Inf predictions")
    for airport_id, airport in enumerate(AIRPORTS):
        idx = np.asarray(arrays["source_index"][arrays["airport_id"] == airport_id])
        expected = set(range(CONTRACT_VAL_ROWS_PER_AIRPORT))
        have = {int(value) for value in idx.tolist()}
        if have != expected:
            missing = sorted(expected - have)
            extra = sorted(have - expected)
            raise RuntimeError(
                f"{airport} source_index must cover 0..{CONTRACT_VAL_ROWS_PER_AIRPORT - 1}; "
                f"missing={missing[:12]} extra={extra[:12]}"
            )


def used_core_parameters(model: AeroWFForecastModel) -> list[nn.Parameter]:
    modules = [
        model.core.encoder_T,
        model.core.encoder_F,
        model.core.hierarchical,
        model.core.exo_encoder,
        model.core.fusion,
    ]
    parameters: list[nn.Parameter] = []
    seen: set[int] = set()
    for module in modules:
        if module is None:
            continue
        for parameter in module.parameters():
            if id(parameter) not in seen:
                parameters.append(parameter)
                seen.add(id(parameter))
    return parameters


def main() -> None:
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.initialization == "pretrained" and not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    for required in (args.data_root / "train", args.data_root / "val", args.contract):
        if not required.exists():
            raise FileNotFoundError(required)

    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    if contract["schema_version"] != "2.0":
        raise RuntimeError("only downstream contract V2 is accepted")
    protocol = contract["forecast"]["primary_protocol"]
    if protocol["id"] != "paper_aligned_wind_forecast":
        raise RuntimeError("formal forecast protocol mismatch")

    sys.path.insert(0, str(args.repo_root))
    os.chdir(args.repo_root)
    from models.AirFM.unified_model import UnifiedSeries2Vec

    set_seed(args.seed)
    device = torch.device("cuda")

    train_parts = [
        AirportForecastDataset(args.data_root / "train", airport) for airport in AIRPORTS
    ]
    val_parts = [
        AirportForecastDataset(args.data_root / "val", airport) for airport in AIRPORTS
    ]
    export_parts = [
        AirportForecastDataset(
            args.data_root / "val",
            airport,
            require_complete_horizons=False,
        )
        for airport in AIRPORTS
    ]
    train_dataset = ConcatDataset(train_parts)
    val_dataset = ConcatDataset(val_parts)
    export_dataset = ConcatDataset(export_parts)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    export_loader = DataLoader(
        export_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    pretrained_checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False
    )
    model_config = dict(pretrained_checkpoint["config"])
    core = UnifiedSeries2Vec(model_config, num_classes=21)
    load_information = None
    if args.initialization == "pretrained":
        load_result = core.load_state_dict(
            pretrained_checkpoint["model_state_dict"], strict=True
        )
        load_information = {
            "missing_keys": list(load_result.missing_keys),
            "unexpected_keys": list(load_result.unexpected_keys),
        }
    model = AeroWFForecastModel(core).to(device)

    core_parameters = used_core_parameters(model)
    head_parameters = list(model.forecast_head.parameters())
    if args.initialization == "pretrained":
        optimizer = torch.optim.AdamW(
            [
                {"params": core_parameters, "lr": args.encoder_lr, "name": "encoder"},
                {"params": head_parameters, "lr": args.head_lr, "name": "head"},
            ],
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = torch.optim.AdamW(
            [
                {"params": core_parameters, "lr": args.scratch_lr, "name": "core"},
                {"params": head_parameters, "lr": args.scratch_lr, "name": "head"},
            ],
            weight_decay=args.weight_decay,
        )

    provenance = {
        "task_contract_version": "2.0",
        "contract_sha256": sha256_file(args.contract),
        "data_release": "AeroWF_v1",
        "task": "forecast",
        "forecast_protocol": "paper_aligned_wind_forecast",
        "initialization": args.initialization,
        "formal_result": bool(args.formal),
        "test_used": False,
        "cross_partition_pairing": False,
        "input_renormalized": False,
        "prediction_clipped": False,
        "metric_scales": ["released_normalized_[0,1]", "inverse_minmax_m/s"],
        "seed": args.seed,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "val_export_samples": len(export_dataset),
        "train_samples_by_airport": {
            part.airport: len(part) for part in train_parts
        },
        "val_samples_by_airport": {part.airport: len(part) for part in val_parts},
        "val_export_samples_by_airport": {
            part.airport: len(part) for part in export_parts
        },
        "horizons_minutes": dict(HORIZONS),
        "targets": list(COMPONENTS),
        "target_shape": [4, 3, 2],
        "batch_size": args.batch_size,
        "max_epochs": args.epochs,
        "patience": args.patience,
        "min_delta": args.min_delta,
        "optimizer": "AdamW",
        "scheduler": None,
        "weight_decay": args.weight_decay,
        "gradient_clip": args.grad_clip,
        "scratch_lr": args.scratch_lr,
        "encoder_lr": args.encoder_lr,
        "head_lr": args.head_lr,
        "selection_metric": "validation pooled normalized MSE",
        "checkpoint_path": str(args.checkpoint) if args.initialization == "pretrained" else None,
        "checkpoint_sha256": (
            sha256_file(args.checkpoint) if args.initialization == "pretrained" else None
        ),
        "checkpoint_load": load_information,
        "repo_commit": git_output(args.repo_root, "rev-parse", "HEAD"),
        "repo_status": git_output(args.repo_root, "status", "--short"),
        "model_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "optimized_core_parameter_count": sum(
            parameter.numel() for parameter in core_parameters
        ),
        "head_parameter_count": sum(parameter.numel() for parameter in head_parameters),
    }
    (args.output_dir / "config.json").write_text(
        json.dumps(json_ready({**vars(args), **provenance}), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    shutil.copy2(Path(__file__).resolve(), args.output_dir / "source.py")
    (args.output_dir / "repo_patch.diff").write_text(
        git_output(args.repo_root, "diff"), encoding="utf-8"
    )

    print(json.dumps(provenance, ensure_ascii=False, indent=2))
    print("Device:", device)
    print("GPU:", torch.cuda.get_device_name(0))
    print(
        "Train batches:",
        len(train_loader),
        "Val batches:",
        len(val_loader),
        "Export batches:",
        len(export_loader),
    )

    history: list[dict[str, Any]] = []
    best_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, device, args.grad_clip)
        val_loss, val_metrics, _ = evaluate(model, val_loader, device)
        elapsed = time.time() - epoch_start
        row = {
            "epoch": epoch,
            "train_mse_norm": train_loss,
            "val_mse_norm": val_loss,
            "val_mae_macro_norm": val_metrics["summary"]["MAE_macro_norm"],
            "val_rmse_macro_norm": val_metrics["summary"]["RMSE_macro_norm"],
            "elapsed_seconds": elapsed,
        }
        history.append(row)
        print(
            f"Epoch {epoch}/{args.epochs} | train_mse={train_loss:.8f} | "
            f"val_mse={val_loss:.8f} | "
            f"val_mae_macro={row['val_mae_macro_norm']:.8f} | "
            f"val_rmse_macro={row['val_rmse_macro_norm']:.8f} | {elapsed:.2f}s",
            flush=True,
        )

        if val_loss < best_loss - args.min_delta:
            best_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_metric": best_loss,
                    "config": json_ready({**vars(args), **provenance}),
                },
                checkpoint_dir / "best_model.pth",
            )
            print("  saved best checkpoint", flush=True)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"Early stopping after {args.patience} unimproved epochs")
                break

    total_seconds = time.time() - start_time
    best_checkpoint = torch.load(
        checkpoint_dir / "best_model.pth", map_location=device, weights_only=False
    )
    model.load_state_dict(best_checkpoint["model_state_dict"], strict=True)
    final_val_loss, final_metrics, _ = evaluate(model, val_loader, device)
    prediction_arrays = collect_full_val_predictions(model, export_loader, device)
    assert_full_val_export_coverage(prediction_arrays)
    np.savez_compressed(args.output_dir / "validation_predictions.npz", **prediction_arrays)

    with (args.output_dir / "history.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)

    result = {
        "status": "success",
        **provenance,
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_val_mse_norm": best_loss,
        "reloaded_best_val_mse_norm": final_val_loss,
        "elapsed_seconds": total_seconds,
        "peak_gpu_memory_mb": torch.cuda.max_memory_allocated() / 1024**2,
        "metrics": final_metrics,
        "best_checkpoint_sha256": sha256_file(checkpoint_dir / "best_model.pth"),
        "validation_predictions_sha256": sha256_file(
            args.output_dir / "validation_predictions.npz"
        ),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
        },
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(json_ready(result), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\nFINAL RESULT")
    print(json.dumps(json_ready(result), ensure_ascii=False, indent=2))
    print("Saved to:", args.output_dir)


if __name__ == "__main__":
    main()
