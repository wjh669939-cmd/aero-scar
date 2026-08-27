#!/usr/bin/env python3
"""AeroWF V2 three-class weather-classification baseline.

The script compares train-from-scratch and unified-pretrained initialization
under one frozen data protocol.  It deliberately never loads weather_code_id
as a feature.  weather_label is loaded only as the supervised target and is
mapped to GOOD/PRECIP/HAZARD according to Downstream Task Contract V2.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from trial_features import (
    FORBIDDEN_INPUT_COLUMNS,
    AllowedContextEncoder,
    build_classification_inputs,
)
from trial_objective import classification_loss, compute_class_weights


AIRPORTS = ("ZBAA", "ZSPD", "ZSSS")
AIRPORT_TO_ID = {name: index for index, name in enumerate(AIRPORTS)}
CLASS_NAMES = ("GOOD", "PRECIP", "HAZARD")
IGNORE_INDEX = -100
UPSTREAM_TO_CLASS = np.asarray(
    [
        IGNORE_INDEX,  # 0 <PAD>
        IGNORE_INDEX,  # 1 <UNK>
        0,             # 2 <GOOD_WX>
        2, 2, 2,       # 3..5 thunderstorm / severe convection
        1, 1, 1, 1, 1, 1, 1, 1,  # 6..13 precipitation
        2, 2, 2, 2, 2, 2, 2,      # 14..20 hazards
    ],
    dtype=np.int64,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--initialization", choices=("scratch", "pretrained"), required=True
    )
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--formal", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-delta", type=float, default=1e-4)
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
    if isinstance(value, (Path, torch.device)):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
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
    except Exception as error:
        return f"UNAVAILABLE: {type(error).__name__}: {error}"


def map_labels(upstream: np.ndarray) -> np.ndarray:
    upstream = np.asarray(upstream, dtype=np.int64)
    if upstream.size and (upstream.min() < 0 or upstream.max() >= len(UPSTREAM_TO_CLASS)):
        raise RuntimeError(
            f"weather_label outside expected 0..20: {upstream.min()}..{upstream.max()}"
        )
    return UPSTREAM_TO_CLASS[upstream]


class AirportClassificationDataset(Dataset):
    """One released airport/split, with only contract-allowed model inputs."""

    def __init__(self, split_root: Path, airport: str):
        self.airport = airport
        self.airport_id = AIRPORT_TO_ID[airport]
        self.root = split_root / airport

        # Model inputs.  weather_code_id is intentionally not opened here.
        self.runway = np.load(self.root / "runway.npy", mmap_mode="r")
        self.runway_mask = np.load(self.root / "runway_mask.npy", mmap_mode="r")
        self.sky_condition = np.load(
            self.root / "exo_cat_sky_condition.npy", mmap_mode="r"
        )
        self.has_gust = np.load(self.root / "exo_cat_has_gust.npy", mmap_mode="r")
        self.is_cavok = np.load(self.root / "exo_cat_is_cavok.npy", mmap_mode="r")
        self.exo_continuous = np.load(
            self.root / "exo_continuous.npy", mmap_mode="r"
        )

        # Target/provenance only; neither is passed to the model.
        upstream_label = np.load(self.root / "weather_label.npy", mmap_mode="r")
        self.label = map_labels(upstream_label)
        self.timestamps = np.load(self.root / "timestamps.npy", mmap_mode="r")
        self.source_index = np.load(self.root / "source_index.npy", mmap_mode="r")

        length = len(self.runway)
        arrays = (
            self.runway_mask,
            self.sky_condition,
            self.has_gust,
            self.is_cavok,
            self.exo_continuous,
            self.label,
            self.timestamps,
            self.source_index,
        )
        if any(len(array) != length for array in arrays):
            raise RuntimeError(f"{self.root}: inconsistent array lengths")
        if self.runway.shape[1:] != (4, 96, 11):
            raise RuntimeError(f"{self.root}: unexpected runway shape {self.runway.shape}")
        if self.runway_mask.shape != (length, 4):
            raise RuntimeError(f"{self.root}: unexpected runway_mask shape")

    def __len__(self) -> int:
        return len(self.runway)

    def __getitem__(self, item: int) -> dict[str, Any]:
        timestamp_ns = int(
            np.asarray(self.timestamps[item]).astype("datetime64[ns]").astype(np.int64)
        )
        sample = build_classification_inputs(
            self.runway[item],
            self.runway_mask[item],
            {
                "sky_condition": int(self.sky_condition[item]),
                "has_gust": int(self.has_gust[item]),
                "is_cavok": int(self.is_cavok[item]),
            },
            self.exo_continuous[item],
        )
        sample["label"] = torch.tensor(int(self.label[item]), dtype=torch.long)
        sample["airport_id"] = torch.tensor(self.airport_id, dtype=torch.long)
        sample["timestamp_ns"] = torch.tensor(timestamp_ns, dtype=torch.long)
        sample["source_index"] = torch.tensor(int(self.source_index[item]), dtype=torch.long)
        return sample


class AeroWFClassificationModel(nn.Module):
    """Airport-level classifier with a leakage-safe context path."""

    def __init__(self, core: nn.Module, sky_known_max: int):
        super().__init__()
        self.core = core
        self.context_encoder = AllowedContextEncoder(sky_known_max, output_dim=32)
        fusion_dim = self.core.fusion.output_dim
        self.classification_head = nn.Sequential(
            nn.LayerNorm(fusion_dim + 32),
            nn.Linear(fusion_dim + 32, len(CLASS_NAMES)),
        )

    def forward(
        self,
        x: torch.Tensor,
        node_mask: torch.Tensor,
        allowed_categorical: dict[str, torch.Tensor],
        exo_continuous: torch.Tensor,
    ) -> torch.Tensor:
        # The original core exogenous encoder is deliberately bypassed because
        # its pretraining config contains significant_wx, a forbidden feature.
        rep_t, rep_f = self.core.encode(
            x,
            exo_categorical=None,
            exo_continuous=None,
            node_mask=node_mask,
        )
        representation = self.core.fusion(rep_t, rep_f)
        context = self.context_encoder(allowed_categorical, exo_continuous)
        return self.classification_head(torch.cat([representation, context], dim=-1))


def move_nested(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: move_nested(item, device) for key, item in value.items()}
    return value


def confusion_matrix_explicit(target: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    matrix = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=np.int64)
    for truth, predicted in zip(target.tolist(), prediction.tolist()):
        matrix[int(truth), int(predicted)] += 1
    return matrix


def metrics_from_confusion(matrix: np.ndarray) -> dict[str, Any]:
    if matrix.shape != (3, 3):
        raise RuntimeError(f"invalid confusion matrix shape {matrix.shape}")
    total = int(matrix.sum())
    if total == 0:
        raise RuntimeError("all classification samples are ignored")

    per_class: dict[str, Any] = {}
    f1_values: list[float] = []
    csi_values: list[float] = []
    for index, name in enumerate(CLASS_NAMES):
        tp = int(matrix[index, index])
        fp = int(matrix[:, index].sum() - tp)
        fn = int(matrix[index, :].sum() - tp)
        support = tp + fn
        if support == 0:
            precision = None if tp + fp == 0 else tp / (tp + fp)
            recall = None
            f1 = None
            csi = None
            false_positive_only = fp > 0
        else:
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / support
            f1 = (
                2.0 * precision * recall / (precision + recall)
                if precision + recall
                else 0.0
            )
            csi = tp / (tp + fp + fn) if tp + fp + fn else 0.0
            false_positive_only = False
            f1_values.append(float(f1))
            csi_values.append(float(csi))
        per_class[name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "csi": csi,
            "support": support,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "false_positive_only": false_positive_only,
        }

    return {
        "accuracy": float(np.trace(matrix) / total),
        "macro_f1": float(np.mean(f1_values)) if f1_values else None,
        "CSI_macro": float(np.mean(csi_values)) if csi_values else None,
        "CSI_GOOD": per_class["GOOD"]["csi"],
        "CSI_PRECIP": per_class["PRECIP"]["csi"],
        "CSI_HAZARD": per_class["HAZARD"]["csi"],
        "evaluable_classes": len(f1_values),
        "per_class": per_class,
        "confusion_matrix_rows_true_columns_predicted": matrix.tolist(),
        "sample_count": total,
    }


def train_label_counts(parts: list[AirportClassificationDataset]) -> np.ndarray:
    counts = np.zeros(len(CLASS_NAMES), dtype=np.int64)
    for part in parts:
        valid = np.asarray(part.label)
        valid = valid[valid != IGNORE_INDEX]
        counts += np.bincount(valid, minlength=len(CLASS_NAMES))
    if np.any(counts == 0):
        raise RuntimeError(f"training class has zero support: {counts.tolist()}")
    return counts


def assert_no_forbidden_input_columns(sample: dict[str, Any]) -> None:
    keys = set(sample)
    nested: set[str] = set()
    for value in sample.values():
        if isinstance(value, dict):
            nested.update(value)
    leaked = [name for name in FORBIDDEN_INPUT_COLUMNS if name in keys or name in nested]
    if leaked:
        raise RuntimeError(f"forbidden classification input columns present: {leaked}")


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    class_weights: torch.Tensor,
    device: torch.device,
    grad_clip: float,
) -> float:
    model.train()
    weighted_loss_sum = 0.0
    valid_count = 0
    for batch_index, batch in enumerate(loader, start=1):
        x = batch["x"].to(device, non_blocking=True)
        node_mask = batch["node_mask"].to(device, non_blocking=True)
        label = batch["label"].to(device, non_blocking=True)
        categorical = move_nested(batch["allowed_categorical"], device)
        continuous = batch["exo_continuous"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x, node_mask, categorical, continuous)
        loss = classification_loss(logits, label, class_weights=class_weights)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite train loss at batch {batch_index}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        count = int((label != IGNORE_INDEX).sum().item())
        weighted_loss_sum += float(loss.item()) * count
        valid_count += count
    return weighted_loss_sum / valid_count


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    class_weights: torch.Tensor,
    device: torch.device,
    collect_predictions: bool = False,
) -> tuple[float, dict[str, Any], dict[str, np.ndarray] | None]:
    model.eval()
    loss_sum = 0.0
    valid_count = 0
    target_parts: list[np.ndarray] = []
    prediction_parts: list[np.ndarray] = []
    collected: dict[str, list[np.ndarray]] | None = None
    if collect_predictions:
        collected = {
            "logits": [],
            "prediction": [],
            "target": [],
            "airport_id": [],
            "timestamp_ns": [],
            "source_index": [],
        }

    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        node_mask = batch["node_mask"].to(device, non_blocking=True)
        label = batch["label"].to(device, non_blocking=True)
        categorical = move_nested(batch["allowed_categorical"], device)
        continuous = batch["exo_continuous"].to(device, non_blocking=True)
        logits = model(x, node_mask, categorical, continuous)
        if not torch.isfinite(logits).all():
            raise RuntimeError("non-finite validation logits")
        prediction = logits.argmax(dim=-1)
        valid = label != IGNORE_INDEX
        count = int(valid.sum().item())
        if count:
            loss = classification_loss(
                logits, label, class_weights=class_weights
            )
            loss_sum += float(loss.item()) * count
            valid_count += count
            target_parts.append(label[valid].cpu().numpy())
            prediction_parts.append(prediction[valid].cpu().numpy())

        if collected is not None:
            collected["logits"].append(logits.cpu().numpy())
            collected["prediction"].append(prediction.cpu().numpy())
            collected["target"].append(label.cpu().numpy())
            for key in ("airport_id", "timestamp_ns", "source_index"):
                collected[key].append(batch[key].cpu().numpy())

    if not target_parts:
        raise RuntimeError("validation contains no evaluable labels")
    target = np.concatenate(target_parts)
    prediction = np.concatenate(prediction_parts)
    matrix = confusion_matrix_explicit(target, prediction)
    metrics = metrics_from_confusion(matrix)

    arrays = None
    if collected is not None:
        arrays = {key: np.concatenate(values) for key, values in collected.items()}
    return loss_sum / valid_count, metrics, arrays


def used_core_parameters(model: AeroWFClassificationModel) -> list[nn.Parameter]:
    modules = [
        model.core.encoder_T,
        model.core.encoder_F,
        model.core.hierarchical,
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


def majority_metrics(
    train_counts: np.ndarray, val_parts: list[AirportClassificationDataset]
) -> dict[str, Any]:
    majority = int(train_counts.argmax())
    target = np.concatenate([np.asarray(part.label) for part in val_parts])
    ignored = int((target == IGNORE_INDEX).sum())
    target = target[target != IGNORE_INDEX]
    prediction = np.full_like(target, majority)
    return {
        "majority_class": majority,
        "majority_class_name": CLASS_NAMES[majority],
        "ignored_samples": ignored,
        **metrics_from_confusion(confusion_matrix_explicit(target, prediction)),
    }


def main() -> None:
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    for required in (args.data_root / "train", args.data_root / "val", args.contract):
        if not required.exists():
            raise FileNotFoundError(required)

    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    if contract["schema_version"] != "2.0":
        raise RuntimeError("only downstream contract V2 is accepted")
    contract_mapping = contract["classification"]["mapping_by_upstream_id"]
    expected_mapping = {
        str(index): (None if value == IGNORE_INDEX else int(value))
        for index, value in enumerate(UPSTREAM_TO_CLASS.tolist())
    }
    if contract_mapping != expected_mapping:
        raise RuntimeError("classification label mapping does not match contract")

    sys.path.insert(0, str(args.repo_root))
    os.chdir(args.repo_root)
    from models.AirFM.unified_model import UnifiedSeries2Vec

    set_seed(args.seed)
    device = torch.device("cuda")
    train_parts = [
        AirportClassificationDataset(args.data_root / "train", airport)
        for airport in AIRPORTS
    ]
    val_parts = [
        AirportClassificationDataset(args.data_root / "val", airport)
        for airport in AIRPORTS
    ]
    train_dataset = ConcatDataset(train_parts)
    val_dataset = ConcatDataset(val_parts)

    counts = train_label_counts(train_parts)
    class_weights_cpu = compute_class_weights(counts)
    for part in (*train_parts, *val_parts):
        if len(part) == 0:
            continue
        assert_no_forbidden_input_columns(part[0])
    class_weights = class_weights_cpu.to(device)
    sky_known_max = max(int(np.asarray(part.sky_condition).max()) for part in train_parts)
    val_sky_unknown_count = sum(
        int((np.asarray(part.sky_condition) > sky_known_max).sum()) for part in val_parts
    )

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
    model = AeroWFClassificationModel(core, sky_known_max).to(device)

    core_parameters = used_core_parameters(model)
    context_head_parameters = list(model.context_encoder.parameters()) + list(
        model.classification_head.parameters()
    )
    if args.initialization == "pretrained":
        optimizer = torch.optim.AdamW(
            [
                {"params": core_parameters, "lr": args.encoder_lr, "name": "encoder"},
                {
                    "params": context_head_parameters,
                    "lr": args.head_lr,
                    "name": "context_and_head",
                },
            ],
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = torch.optim.AdamW(
            [
                {"params": core_parameters, "lr": args.scratch_lr, "name": "core"},
                {
                    "params": context_head_parameters,
                    "lr": args.scratch_lr,
                    "name": "context_and_head",
                },
            ],
            weight_decay=args.weight_decay,
        )

    ignored_train = sum(int((np.asarray(part.label) == IGNORE_INDEX).sum()) for part in train_parts)
    ignored_val = sum(int((np.asarray(part.label) == IGNORE_INDEX).sum()) for part in val_parts)
    majority = majority_metrics(counts, val_parts)
    provenance = {
        "task_contract_version": "2.0",
        "contract_sha256": sha256_file(args.contract),
        "data_release": "AeroWF_v1",
        "task": "classification",
        "forecast_protocol": None,
        "initialization": args.initialization,
        "formal_result": bool(args.formal),
        "result_scope": "validation_candidate_not_sealed_test",
        "test_used": False,
        "input_renormalized": False,
        "seed": args.seed,
        "classes": {str(index): name for index, name in enumerate(CLASS_NAMES)},
        "ignore_index": IGNORE_INDEX,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "train_samples_by_airport": {part.airport: len(part) for part in train_parts},
        "val_samples_by_airport": {part.airport: len(part) for part in val_parts},
        "train_class_counts": {
            CLASS_NAMES[index]: int(counts[index]) for index in range(3)
        },
        "train_class_weights": {
            CLASS_NAMES[index]: float(class_weights_cpu[index]) for index in range(3)
        },
        "ignored_train_samples": ignored_train,
        "ignored_val_samples": ignored_val,
        "batch_size": args.batch_size,
        "max_epochs": args.epochs,
        "patience": args.patience,
        "min_delta": args.min_delta,
        "optimizer": "AdamW",
        "scheduler": None,
        "loss": "train-only balanced weighted cross entropy",
        "weight_decay": args.weight_decay,
        "gradient_clip": args.grad_clip,
        "scratch_lr": args.scratch_lr,
        "encoder_lr": args.encoder_lr,
        "head_lr": args.head_lr,
        "selection_metric": "validation Macro-F1",
        "zero_support_policy": "contract_v2_exclude_from_macro_and_report_fp",
        "checkpoint_path": (
            str(args.checkpoint) if args.initialization == "pretrained" else None
        ),
        "checkpoint_sha256": (
            sha256_file(args.checkpoint)
            if args.initialization == "pretrained"
            else None
        ),
        "checkpoint_load": load_information,
        "sky_known_max_train_only": sky_known_max,
        "validation_unknown_sky_count": val_sky_unknown_count,
        "leakage_controls": {
            "weather_code_id_loaded": False,
            "weather_label_used_as_input": False,
            "weather_label_used_as_target_only": True,
            "significant_wx_used": False,
            "core_pretraining_exogenous_encoder_bypassed": True,
            "allowed_categorical_inputs": [
                "sky_condition",
                "has_gust",
                "is_cavok",
            ],
            "allowed_continuous_inputs": [
                "visibility",
                "cloud_height",
                "gust_speed",
            ],
        },
        "repo_commit": git_output(args.repo_root, "rev-parse", "HEAD"),
        "repo_status": git_output(args.repo_root, "status", "--short"),
        "model_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "optimized_core_parameter_count": sum(
            parameter.numel() for parameter in core_parameters
        ),
        "context_head_parameter_count": sum(
            parameter.numel() for parameter in context_head_parameters
        ),
        "majority_validation_baseline": majority,
    }

    (args.output_dir / "config.json").write_text(
        json.dumps(json_ready({**vars(args), **provenance}), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    shutil.copy2(Path(__file__).resolve(), args.output_dir / "source.py")
    (args.output_dir / "repo_patch.diff").write_text(
        git_output(args.repo_root, "diff"), encoding="utf-8"
    )

    print(json.dumps(json_ready(provenance), ensure_ascii=False, indent=2))
    print("Device:", device)
    print("GPU:", torch.cuda.get_device_name(0))
    print("Train batches:", len(train_loader), "Val batches:", len(val_loader))

    history: list[dict[str, Any]] = []
    best_macro_f1 = -float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        train_loss = train_epoch(
            model, train_loader, optimizer, class_weights, device, args.grad_clip
        )
        val_loss, val_metrics, _ = evaluate(
            model, val_loader, class_weights, device
        )
        macro_f1 = val_metrics["macro_f1"]
        if macro_f1 is None:
            raise RuntimeError("validation Macro-F1 is NA")
        elapsed = time.time() - epoch_start
        row = {
            "epoch": epoch,
            "train_weighted_ce": train_loss,
            "val_weighted_ce": val_loss,
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": macro_f1,
            "val_CSI_macro": val_metrics["CSI_macro"],
            "elapsed_seconds": elapsed,
        }
        history.append(row)
        print(
            f"Epoch {epoch}/{args.epochs} | train_ce={train_loss:.8f} | "
            f"val_ce={val_loss:.8f} | accuracy={row['val_accuracy']:.8f} | "
            f"macro_f1={macro_f1:.8f} | CSI={row['val_CSI_macro']:.8f} | "
            f"{elapsed:.2f}s",
            flush=True,
        )

        if macro_f1 > best_macro_f1 + args.min_delta:
            best_macro_f1 = float(macro_f1)
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_metric": best_macro_f1,
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
    final_val_loss, final_metrics, arrays = evaluate(
        model, val_loader, class_weights, device, collect_predictions=True
    )
    assert arrays is not None
    np.savez_compressed(args.output_dir / "validation_predictions.npz", **arrays)

    per_airport: dict[str, Any] = {}
    for part in val_parts:
        loader = DataLoader(
            part,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        airport_loss, airport_metrics, _ = evaluate(
            model, loader, class_weights, device
        )
        per_airport[part.airport] = {
            "weighted_ce": airport_loss,
            **airport_metrics,
        }

    with (args.output_dir / "history.csv").open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)

    result = {
        "status": "success",
        **provenance,
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_macro_f1,
        "reloaded_best_val_weighted_ce": final_val_loss,
        "elapsed_seconds": total_seconds,
        "peak_gpu_memory_mb": torch.cuda.max_memory_allocated() / 1024**2,
        "metrics": final_metrics,
        "metrics_by_airport": per_airport,
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
        json.dumps(json_ready(result), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("\nFINAL RESULT")
    print(json.dumps(json_ready(result), ensure_ascii=False, indent=2))
    print("Saved to:", args.output_dir)


if __name__ == "__main__":
    main()
