from pathlib import Path
import hashlib
import json
import random
import subprocess

import numpy as np
import torch

from models.AirFM.unified_model import UnifiedSeries2Vec


DATA_ROOT = Path("subset/finetune")
CHECKPOINT_PATH = Path(
    "results/aerowf_official_subset_1epoch/checkpoints/best_model.pth"
)
OUTPUT_ROOT = Path(
    "results/aerowf_official_subset_fixedmask_eval_v1"
)

BATCH_SIZE = 16
MASK_SEED_BASE = 20260822
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def get_git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except Exception:
        return "unknown"


# ============================================================
# 1. 使用训练集计算与训练脚本完全相同的全局 Min-Max
# ============================================================

files = sorted(DATA_ROOT.glob("*_train.npy"))

if not files:
    raise FileNotFoundError(
        f"没有在 {DATA_ROOT} 找到 *_train.npy"
    )

feature_min = np.full(11, np.inf, dtype=np.float32)
feature_max = np.full(11, -np.inf, dtype=np.float32)

for path in files:
    data = np.load(path, allow_pickle=True).item()
    train_x = data["train_runway"]

    feature_min = np.minimum(
        feature_min,
        train_x.min(axis=(0, 1, 2)),
    )
    feature_max = np.maximum(
        feature_max,
        train_x.max(axis=(0, 1, 2)),
    )

denominator = np.maximum(
    feature_max - feature_min,
    1e-8,
)


# ============================================================
# 2. 加载 checkpoint 和官方模型
# ============================================================

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location="cpu",
    weights_only=False,
)

config = dict(checkpoint["config"])
config["device"] = DEVICE

model = UnifiedSeries2Vec(
    config,
    num_classes=3,
)

load_result = model.load_state_dict(
    checkpoint["model_state_dict"],
    strict=True,
)

model = model.to(DEVICE)
model.eval()

print("Checkpoint:", CHECKPOINT_PATH)
print("Checkpoint epoch:", checkpoint["epoch"])
print("Checkpoint best metric:", checkpoint["best_metric"])
print("Device:", DEVICE)
print("Missing keys:", load_result.missing_keys)
print("Unexpected keys:", load_result.unexpected_keys)
print(
    "Parameter count:",
    sum(parameter.numel() for parameter in model.parameters()),
)


# ============================================================
# 3. 固定 mask，在有效跑道、遮挡时刻和全部特征上评测
# ============================================================

per_airport = {}

overall_squared_sum = 0.0
overall_absolute_sum = 0.0
overall_element_count = 0
overall_model_loss_sum = 0.0
overall_samples = 0

torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None

with torch.no_grad():
    for airport_index, path in enumerate(files):
        airport = path.stem.replace("_train", "")
        data = np.load(path, allow_pickle=True).item()

        raw_x = data["val_runway"].astype(np.float32)
        num_samples, num_runways, time_steps, num_features = raw_x.shape

        scaled_x_unclipped = (
            raw_x - feature_min.reshape(1, 1, 1, -1)
        ) / denominator.reshape(1, 1, 1, -1)

        clip_count = int(
            (
                (scaled_x_unclipped < 0.0)
                | (scaled_x_unclipped > 1.0)
            ).sum()
        )
        clip_fraction = (
            clip_count / scaled_x_unclipped.size
            if scaled_x_unclipped.size
            else 0.0
        )

        scaled_x = np.clip(
            scaled_x_unclipped,
            0.0,
            1.0,
        ).astype(np.float32)

        padded_x = np.zeros(
            (num_samples, 4, time_steps, num_features),
            dtype=np.float32,
        )
        padded_x[:, :num_runways] = scaled_x

        runway_mask = np.zeros(
            (num_samples, 4),
            dtype=bool,
        )
        runway_mask[:, :num_runways] = True

        categorical = data["val_exo_categorical"]

        weather_code = np.asarray(
            categorical["weather_code_id"],
            dtype=np.int64,
        )

        significant_wx = (
            weather_code != 2
        ).astype(np.int64)

        sky_condition = np.asarray(
            categorical["sky_condition"],
            dtype=np.int64,
        )
        has_gust = np.asarray(
            categorical["has_gust"],
            dtype=np.int64,
        )
        is_cavok = np.asarray(
            categorical["is_cavok"],
            dtype=np.int64,
        )

        continuous = np.asarray(
            data["val_exo_continuous"],
            dtype=np.float32,
        )

        # 每个机场使用固定且独立的 mask 随机种子
        airport_mask_seed = MASK_SEED_BASE + airport_index
        random.seed(airport_mask_seed)
        np.random.seed(airport_mask_seed)
        torch.manual_seed(airport_mask_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(airport_mask_seed)

        squared_sum = 0.0
        absolute_sum = 0.0
        element_count = 0
        model_loss_sum = 0.0

        for start in range(0, num_samples, BATCH_SIZE):
            end = min(start + BATCH_SIZE, num_samples)

            x_batch = torch.from_numpy(
                padded_x[start:end]
            ).to(DEVICE)

            node_mask_batch = torch.from_numpy(
                runway_mask[start:end]
            ).to(DEVICE)

            exo_categorical = {
                "significant_wx": torch.from_numpy(
                    significant_wx[start:end]
                ).long().to(DEVICE),
                "sky_condition": torch.from_numpy(
                    sky_condition[start:end]
                ).long().to(DEVICE),
                "has_gust": torch.from_numpy(
                    has_gust[start:end]
                ).long().to(DEVICE),
                "is_cavok": torch.from_numpy(
                    is_cavok[start:end]
                ).long().to(DEVICE),
            }

            exo_continuous = {
                "visibility": torch.from_numpy(
                    continuous[start:end, 0]
                ).float().to(DEVICE),
                "cloud_height": torch.from_numpy(
                    continuous[start:end, 1]
                ).float().to(DEVICE),
                "gust_speed": torch.from_numpy(
                    continuous[start:end, 2]
                ).float().to(DEVICE),
            }

            loss, loss_information, outputs = model(
                x_batch,
                mode="masked_recon",
                exo_categorical=exo_categorical,
                exo_continuous=exo_continuous,
                node_mask=node_mask_batch,
                return_all=True,
            )

            target = outputs["original"]
            prediction = outputs["reconstructed"]
            temporal_mask = outputs["mask"]

            current_batch = end - start
            current_nodes = x_batch.shape[1]

            # [B,T] -> [B*N,T]
            temporal_mask = temporal_mask.repeat_interleave(
                current_nodes,
                dim=0,
            ).bool()

            # [B,N] -> [B*N]
            valid_nodes = node_mask_batch.reshape(-1).bool()

            # 扩展到全部特征通道：[B*N,C,T]
            element_mask = temporal_mask.unsqueeze(1).expand(
                -1,
                prediction.shape[1],
                -1,
            )
            element_mask = (
                element_mask
                & valid_nodes.view(-1, 1, 1)
            )

            error = prediction - target
            selected_error = error[element_mask]

            batch_element_count = int(
                selected_error.numel()
            )

            if batch_element_count == 0:
                continue

            batch_squared_sum = float(
                selected_error.double().square().sum().item()
            )
            batch_absolute_sum = float(
                selected_error.double().abs().sum().item()
            )

            squared_sum += batch_squared_sum
            absolute_sum += batch_absolute_sum
            element_count += batch_element_count

            # 官方修正后的 masked_mse_loss 应与手动 MSE 一致
            model_loss_sum += (
                float(loss.item()) * batch_element_count
            )

        airport_mse = squared_sum / element_count
        airport_mae = absolute_sum / element_count
        airport_model_mse = model_loss_sum / element_count

        per_airport[airport] = {
            "samples": int(num_samples),
            "num_runways": int(num_runways),
            "mask_seed": int(airport_mask_seed),
            "mask_ratio": float(config["mask_ratio"]),
            "evaluated_elements": int(element_count),
            "mse": airport_mse,
            "mae": airport_mae,
            "model_reported_mse": airport_model_mse,
            "mse_consistency_difference": abs(
                airport_mse - airport_model_mse
            ),
            "input_clip_fraction": float(clip_fraction),
        }

        overall_squared_sum += squared_sum
        overall_absolute_sum += absolute_sum
        overall_element_count += element_count
        overall_model_loss_sum += model_loss_sum
        overall_samples += num_samples

        print("\n" + "=" * 70)
        print("Airport:", airport)
        print("Samples:", num_samples)
        print("Valid runways:", num_runways)
        print("Mask seed:", airport_mask_seed)
        print("Evaluated elements:", element_count)
        print("Fixed-mask MSE:", airport_mse)
        print("Fixed-mask MAE:", airport_mae)
        print("Model-reported MSE:", airport_model_mse)
        print("Input clip fraction:", clip_fraction)


overall = {
    "samples": int(overall_samples),
    "evaluated_elements": int(overall_element_count),
    "mse": overall_squared_sum / overall_element_count,
    "mae": overall_absolute_sum / overall_element_count,
    "model_reported_mse": (
        overall_model_loss_sum / overall_element_count
    ),
    "test_used": False,
    "prediction_clipped": False,
    "mask_ratio": float(config["mask_ratio"]),
    "mask_strategy": config["random_mask_strategy"],
}

overall["mse_consistency_difference"] = abs(
    overall["mse"] - overall["model_reported_mse"]
)

metrics = {
    "evaluation_name": (
        "AeroWF official subset fixed-mask validation v1"
    ),
    "checkpoint": str(CHECKPOINT_PATH),
    "checkpoint_sha256": sha256_file(CHECKPOINT_PATH),
    "checkpoint_epoch": int(checkpoint["epoch"]),
    "checkpoint_best_random_mask_val_loss": float(
        checkpoint["best_metric"]
    ),
    "git_commit": get_git_commit(),
    "split": "val",
    "test_used": False,
    "fixed_mask": True,
    "mask_seed_base": MASK_SEED_BASE,
    "overall": overall,
    "per_airport": per_airport,
    "feature_min": feature_min.tolist(),
    "feature_max": feature_max.tolist(),
    "data_sha256": {
        path.name: sha256_file(path)
        for path in files
    },
    "peak_gpu_memory_mb": (
        torch.cuda.max_memory_allocated() / 1024**2
        if torch.cuda.is_available()
        else 0.0
    ),
}

with open(
    OUTPUT_ROOT / "metrics.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        metrics,
        file,
        ensure_ascii=False,
        indent=2,
    )

evaluation_config = {
    "split": "val",
    "test_used": False,
    "batch_size": BATCH_SIZE,
    "mask_ratio": float(config["mask_ratio"]),
    "mask_strategy": config["random_mask_strategy"],
    "causal_probability": float(config["causal_prob"]),
    "mask_seed_base": MASK_SEED_BASE,
    "metric_scope": (
        "valid runways × masked timesteps × all 11 features"
    ),
    "prediction_clipped": False,
}

with open(
    OUTPUT_ROOT / "evaluation_config.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        evaluation_config,
        file,
        ensure_ascii=False,
        indent=2,
    )

# 保存当前官方代码修复记录
git_diff = subprocess.run(
    ["git", "diff", "--no-ext-diff"],
    capture_output=True,
    text=True,
).stdout

with open(
    OUTPUT_ROOT / "source_patch.diff",
    "w",
    encoding="utf-8",
) as file:
    file.write(git_diff)

print("\n" + "=" * 70)
print("FINAL FIXED-MASK VALIDATION")
print(json.dumps(overall, ensure_ascii=False, indent=2))
print("\nSaved to:", OUTPUT_ROOT)
