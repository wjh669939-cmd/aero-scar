from pathlib import Path
import json
import random
import time

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from models.AirFM.unified_model import UnifiedSeries2Vec
from models.AirFM.unified_trainer import UnifiedTrainer


SEED = 42
DATA_ROOT = Path("data/AeroWF/processed")
OUTPUT_ROOT = Path("results/aerowf_official_full_scratch_30epoch_v1")

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
(OUTPUT_ROOT / "checkpoints").mkdir(exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

FILES = sorted(DATA_ROOT.glob("*_train.npy"))

# 仅使用训练集计算全局Min-Max
feature_min = np.full(11, np.inf, dtype=np.float32)
feature_max = np.full(11, -np.inf, dtype=np.float32)
sky_max = 0

for path in FILES:
    data = np.load(path, allow_pickle=True).item()
    x = data["train_runway"]

    feature_min = np.minimum(
        feature_min,
        x.min(axis=(0, 1, 2)),
    )
    feature_max = np.maximum(
        feature_max,
        x.max(axis=(0, 1, 2)),
    )

    sky = np.asarray(
        data["train_exo_categorical"]["sky_condition"]
    )
    sky_max = max(sky_max, int(sky.max()))

denominator = np.maximum(
    feature_max - feature_min,
    1e-8,
)


class AeroSubsetDataset(Dataset):
    def __init__(self, split):
        all_x = []
        all_masks = []
        all_labels = []

        all_significant_wx = []
        all_sky = []
        all_has_gust = []
        all_is_cavok = []
        all_continuous = []

        clipped_count = 0
        total_count = 0

        for path in FILES:
            data = np.load(path, allow_pickle=True).item()

            raw_x = data[f"{split}_runway"].astype(np.float32)
            batch_size, num_runways, time_steps, features = raw_x.shape

            scaled_x = (
                raw_x - feature_min.reshape(1, 1, 1, -1)
            ) / denominator.reshape(1, 1, 1, -1)

            clipped_count += int(
                ((scaled_x < 0.0) | (scaled_x > 1.0)).sum()
            )
            total_count += int(scaled_x.size)

            # 官方数据说明中的运行时Min-Max及clip
            scaled_x = np.clip(
                scaled_x,
                0.0,
                1.0,
            ).astype(np.float32)

            padded_x = np.zeros(
                (batch_size, 4, time_steps, features),
                dtype=np.float32,
            )
            padded_x[:, :num_runways] = scaled_x

            runway_mask = np.zeros(
                (batch_size, 4),
                dtype=bool,
            )
            runway_mask[:, :num_runways] = True

            categorical = data[f"{split}_exo_categorical"]

            weather_code = np.asarray(
                categorical["weather_code_id"],
                dtype=np.int64,
            )

            # 按论文Table 1，将天气代码简化成显著天气二值标志
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
                data[f"{split}_exo_continuous"],
                dtype=np.float32,
            )

            labels = np.asarray(
                data[f"{split}_weather_label"],
                dtype=np.int64,
            )

            all_x.append(padded_x)
            all_masks.append(runway_mask)
            all_labels.append(labels)

            all_significant_wx.append(significant_wx)
            all_sky.append(sky_condition)
            all_has_gust.append(has_gust)
            all_is_cavok.append(is_cavok)
            all_continuous.append(continuous)

        self.x = np.concatenate(all_x)
        self.runway_mask = np.concatenate(all_masks)
        self.labels = np.concatenate(all_labels)

        self.significant_wx = np.concatenate(
            all_significant_wx
        )
        self.sky = np.concatenate(all_sky)
        self.has_gust = np.concatenate(all_has_gust)
        self.is_cavok = np.concatenate(all_is_cavok)
        self.continuous = np.concatenate(all_continuous)

        self.clip_fraction = (
            clipped_count / total_count
            if total_count else 0.0
        )

        print(
            f"{split}: samples={len(self.x)}, "
            f"shape={self.x.shape}, "
            f"clip_fraction={self.clip_fraction:.8f}"
        )

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return {
            "x": torch.from_numpy(self.x[index]),
            "runway_mask": torch.from_numpy(
                self.runway_mask[index]
            ),
            # masked_recon不使用标签，但Trainer要求该字段存在
            "label_21": torch.tensor(
                self.labels[index],
                dtype=torch.long,
            ),
            "exo_categorical": {
                "significant_wx": torch.tensor(
                    self.significant_wx[index],
                    dtype=torch.long,
                ),
                "sky_condition": torch.tensor(
                    self.sky[index],
                    dtype=torch.long,
                ),
                "has_gust": torch.tensor(
                    self.has_gust[index],
                    dtype=torch.long,
                ),
                "is_cavok": torch.tensor(
                    self.is_cavok[index],
                    dtype=torch.long,
                ),
            },
            "exo_continuous": {
                "visibility": torch.tensor(
                    self.continuous[index, 0],
                    dtype=torch.float32,
                ),
                "cloud_height": torch.tensor(
                    self.continuous[index, 1],
                    dtype=torch.float32,
                ),
                "gust_speed": torch.tensor(
                    self.continuous[index, 2],
                    dtype=torch.float32,
                ),
            },
        }


train_dataset = AeroSubsetDataset("train")
val_dataset = AeroSubsetDataset("val")

generator = torch.Generator()
generator.manual_seed(SEED)

train_loader = DataLoader(
    train_dataset,
    batch_size=128,
    shuffle=True,
    num_workers=0,
    pin_memory=True,
    generator=generator,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=128,
    shuffle=False,
    num_workers=0,
    pin_memory=True,
)

device = torch.device("cuda")

config = {
    "Data_shape": (128, 4, 96, 11),
    "N_max": 4,

    "emb_size": 128,
    "rep_size": 256,
    "num_heads": 8,
    "dim_ff": 256,
    "dropout": 0.2,

    "encoder_num_heads": 4,
    "encoder_num_layers": 3,
    "encoder_dim_ff": 512,
    "frets_num_layers": 2,

    "use_simple_gnn": True,
    "gnn_hidden": 128,
    "gnn_layers": 2,

    "fusion_type": "residual_add",
    "use_frequency": True,
    "use_hierarchy": True,

    "use_masked_recon": True,
    "mask_ratio": 0.25,
    "random_mask_strategy": "random",
    "causal_mask_strategy": "last",
    "causal_prob": 0.0,

    "exo_config": {
        "categorical": {
            "significant_wx": {
                "vocab_size": 2,
            },
            "sky_condition": {
                "vocab_size": sky_max + 1,
            },
            "has_gust": {
                "vocab_size": 2,
            },
            "is_cavok": {
                "vocab_size": 2,
            },
        },
        "continuous": [
            "visibility",
            "cloud_height",
            "gust_speed",
        ],
    },

    "task_type": "masked_recon",
    "training_mode": "masked_recon",

    "epochs": 30,
    "batch_size": 128,
    "lr": 3e-4,
    "weight_decay": 1e-4,
    "warmup_epochs": 5,
    "patience": 10,
    "min_delta": 1e-4,
    "grad_clip": 3.0,

    "device": device,
    "seed": SEED,

    "save_dir": str(OUTPUT_ROOT / "checkpoints"),
    "output_dir": str(OUTPUT_ROOT),
}

with open(
    OUTPUT_ROOT / "config.json",
    "w",
    encoding="utf-8",
) as file:
    serializable_config = {
        key: str(value) if key == "device" else value
        for key, value in config.items()
    }
    json.dump(
        serializable_config,
        file,
        ensure_ascii=False,
        indent=2,
    )

model = UnifiedSeries2Vec(
    config,
    num_classes=3,
)

print(
    "Parameter count:",
    sum(parameter.numel() for parameter in model.parameters()),
)

torch.cuda.reset_peak_memory_stats()
start_time = time.time()

trainer = UnifiedTrainer(
    model,
    config,
    device=device,
)

metrics = trainer.train(
    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=None,
    num_epochs=30,
)

elapsed_seconds = time.time() - start_time

metrics["elapsed_seconds"] = elapsed_seconds
metrics["peak_gpu_memory_mb"] = (
    torch.cuda.max_memory_allocated() / 1024**2
)
metrics["test_used"] = False
metrics["train_clip_fraction"] = train_dataset.clip_fraction
metrics["val_clip_fraction"] = val_dataset.clip_fraction

with open(
    OUTPUT_ROOT / "metrics.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        dict(metrics),
        file,
        ensure_ascii=False,
        indent=2,
        default=float,
    )

print("\nFinal metrics:")
print(
    json.dumps(
        dict(metrics),
        ensure_ascii=False,
        indent=2,
        default=float,
    )
)
