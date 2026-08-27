from pathlib import Path
from bisect import bisect_right
import json
import logging
import random
import shutil
import time

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from models.AirFM.unified_model import UnifiedSeries2Vec
from models.AirFM.unified_trainer import UnifiedTrainer


# ============================================================
# 1. 实验设置
# ============================================================

SEED = 42
BATCH_SIZE = 128
AIRPORTS = ["ZBAA", "ZSPD", "ZSSS"]

DATA_ROOT = Path(
    "/root/autodl-tmp/aerowf_delivery/v1/"
    "extracted/AeroWF_v1_MODEL_TRAINING/"
    "release_v1/pretrain"
)

OUTPUT_ROOT = Path(
    "results/aerowf_unified_pretrain_full_1epoch_smoke_v2"
)

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
(OUTPUT_ROOT / "checkpoints").mkdir(exist_ok=True)

if (OUTPUT_ROOT / "metrics.json").exists():
    raise RuntimeError(
        f"{OUTPUT_ROOT}/metrics.json 已存在，请勿覆盖已有实验"
    )

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

device = torch.device("cuda")

if not torch.cuda.is_available():
    raise RuntimeError("CUDA不可用")


# ============================================================
# 2. 冻结数据读取
# ============================================================

class FrozenAeroWFDataset(Dataset):
    def __init__(self, split):
        self.split = split
        self.groups = []
        self.ends = []
        self.total_samples = 0

        for airport in AIRPORTS:
            airport_root = DATA_ROOT / split / airport

            required = [
                "runway.npy",
                "runway_mask.npy",
                "exo_cat_weather_code_id.npy",
                "exo_cat_sky_condition.npy",
                "exo_cat_has_gust.npy",
                "exo_cat_is_cavok.npy",
                "exo_continuous.npy",
            ]

            for filename in required:
                if not (airport_root / filename).exists():
                    raise FileNotFoundError(
                        airport_root / filename
                    )

            group = {
                "airport": airport,
                "runway": np.load(
                    airport_root / "runway.npy",
                    mmap_mode="r",
                ),
                "runway_mask": np.load(
                    airport_root / "runway_mask.npy",
                    mmap_mode="r",
                ),
                "weather_code": np.load(
                    airport_root / "exo_cat_weather_code_id.npy",
                    mmap_mode="r",
                ),
                "sky_condition": np.load(
                    airport_root / "exo_cat_sky_condition.npy",
                    mmap_mode="r",
                ),
                "has_gust": np.load(
                    airport_root / "exo_cat_has_gust.npy",
                    mmap_mode="r",
                ),
                "is_cavok": np.load(
                    airport_root / "exo_cat_is_cavok.npy",
                    mmap_mode="r",
                ),
                "exo_continuous": np.load(
                    airport_root / "exo_continuous.npy",
                    mmap_mode="r",
                ),
            }

            sample_count = len(group["runway"])

            for key, value in group.items():
                if key == "airport":
                    continue
                if len(value) != sample_count:
                    raise RuntimeError(
                        f"{airport}/{split}/{key}样本数不一致"
                    )

            if group["runway"].shape[1:] != (4, 96, 11):
                raise RuntimeError(
                    f"{airport}/{split}形状异常："
                    f"{group['runway'].shape}"
                )

            self.total_samples += sample_count
            self.ends.append(self.total_samples)
            self.groups.append(group)

            print(
                f"{split}/{airport}: "
                f"samples={sample_count}, "
                f"shape={group['runway'].shape}"
            )

        print(
            f"{split} total samples: {self.total_samples}"
        )

    def __len__(self):
        return self.total_samples

    def __getitem__(self, index):
        group_index = bisect_right(self.ends, index)

        previous_end = (
            0 if group_index == 0
            else self.ends[group_index - 1]
        )

        local_index = index - previous_end
        group = self.groups[group_index]

        x = np.array(
            group["runway"][local_index],
            dtype=np.float32,
            copy=True,
        )

        runway_mask = np.array(
            group["runway_mask"][local_index],
            dtype=bool,
            copy=True,
        )

        weather_code = int(
            group["weather_code"][local_index]
        )

        sky_condition = int(
            group["sky_condition"][local_index]
        )

        has_gust = int(
            group["has_gust"][local_index]
        )

        is_cavok = int(
            group["is_cavok"][local_index]
        )

        exo_continuous = np.array(
            group["exo_continuous"][local_index],
            dtype=np.float32,
            copy=True,
        )

        return {
            "x": torch.from_numpy(x),
            "runway_mask": torch.from_numpy(runway_mask),

            # 预训练不使用天气标签，仅用于满足Trainer接口
            "label_21": torch.tensor(
                0,
                dtype=torch.long,
            ),

            "exo_categorical": {
                "significant_wx": torch.tensor(
                    int(weather_code != 2),
                    dtype=torch.long,
                ),
                "sky_condition": torch.tensor(
                    sky_condition,
                    dtype=torch.long,
                ),
                "has_gust": torch.tensor(
                    has_gust,
                    dtype=torch.long,
                ),
                "is_cavok": torch.tensor(
                    is_cavok,
                    dtype=torch.long,
                ),
            },

            "exo_continuous": {
                "visibility": torch.tensor(
                    exo_continuous[0],
                    dtype=torch.float32,
                ),
                "cloud_height": torch.tensor(
                    exo_continuous[1],
                    dtype=torch.float32,
                ),
                "gust_speed": torch.tensor(
                    exo_continuous[2],
                    dtype=torch.float32,
                ),
            },
        }


train_dataset = FrozenAeroWFDataset("train")
val_dataset = FrozenAeroWFDataset("val")

if len(train_dataset) != 97914:
    raise RuntimeError(
        f"训练样本数异常：{len(train_dataset)}"
    )

if len(val_dataset) != 20982:
    raise RuntimeError(
        f"验证样本数异常：{len(val_dataset)}"
    )


# ============================================================
# 3. 确定类别字段范围
# ============================================================

sky_max = 0

for dataset in [train_dataset, val_dataset]:
    for group in dataset.groups:
        sky_max = max(
            sky_max,
            int(np.max(group["sky_condition"])),
        )

print("Sky condition maximum:", sky_max)


# ============================================================
# 4. DataLoader
# ============================================================

generator = torch.Generator()
generator.manual_seed(SEED)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=True,
    drop_last=False,
    generator=generator,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=True,
    drop_last=False,
)


# ============================================================
# 5. 模型和训练配置
# ============================================================

config = {
    "Data_shape": (
        BATCH_SIZE,
        4,
        96,
        11,
    ),
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
    "causal_prob": 0.5,

    "proj_hidden_dim": 256,
    "proj_output_dim": 128,

    "lambda_recon": 1.0,
    "lambda_contrast": 0.5,

    "ema_momentum": 0.99,
    "warmup_batches": 10,

    "sdtw_gamma": 0.1,
    "softdtw_pair_chunk_size": 256,

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

    "task_type": "unified_pretrain",
    "training_mode": "unified_pretrain",

    "epochs": 1,
    "batch_size": BATCH_SIZE,

    "optimizer": "AdamW",
    "lr": 3e-4,
    "weight_decay": 1e-4,
    "min_lr": 1e-6,
    "warmup_epochs": 5,

    "patience": 10,
    "min_delta": 1e-4,
    "grad_clip": 3.0,

    "device": device,
    "seed": SEED,

    "save_dir": str(
        OUTPUT_ROOT / "checkpoints"
    ),
    "output_dir": str(OUTPUT_ROOT),
}


def to_jsonable(value):
    if isinstance(value, dict):
        return {
            str(key): to_jsonable(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            to_jsonable(item)
            for item in value
        ]

    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.item()
        return value.detach().cpu().tolist()

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, torch.device):
        return str(value)

    return value


with open(
    OUTPUT_ROOT / "config.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        to_jsonable(config),
        file,
        ensure_ascii=False,
        indent=2,
    )

shutil.copy2(
    __file__,
    OUTPUT_ROOT / Path(__file__).name,
)


# ============================================================
# 6. 从头初始化模型
# ============================================================

model = UnifiedSeries2Vec(
    config,
    num_classes=21,
)

parameter_count = sum(
    parameter.numel()
    for parameter in model.parameters()
)

print("Device:", device)
print("GPU:", torch.cuda.get_device_name(0))
print("Parameter count:", parameter_count)
print("Initialization: scratch")
print("Test used: False")

trainer = UnifiedTrainer(
    model,
    config,
    device=device,
)


# ============================================================
# 7. Trainer接口预检查
# ============================================================

probe_batch = next(iter(train_loader))
parsed = trainer._parse_batch(probe_batch)

print("Parsed batch input shape:", parsed[0].shape)
print("Parsed node mask shape:", parsed[3].shape)
print(
    "Parsed exogenous categorical:",
    parsed[4] is not None,
)
print(
    "Parsed exogenous continuous:",
    parsed[5] is not None,
)

del probe_batch
del parsed


# ============================================================
# 8. 完整数据1 epoch训练
# ============================================================

torch.cuda.reset_peak_memory_stats()
start_time = time.time()

trainer_metrics = trainer.train(
    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=None,
    num_epochs=1,
)

elapsed_seconds = time.time() - start_time
peak_gpu_memory_mb = (
    torch.cuda.max_memory_allocated()
    / 1024 ** 2
)

best_checkpoint = (
    OUTPUT_ROOT
    / "checkpoints"
    / "best_model.pth"
)

result = {
    "status": "success",
    "experiment_role": (
        "full_frozen_data_unified_pretrain_"
        "one_epoch_smoke"
    ),
    "formal_result": False,
    "initialization": "scratch",
    "seed": SEED,
    "airports": AIRPORTS,
    "train_samples": len(train_dataset),
    "val_samples": len(val_dataset),
    "test_used": False,
    "weather_label_used": False,
    "input_renormalized": False,
    "batch_size": BATCH_SIZE,
    "epochs": 1,
    "mask_ratio": 0.25,
    "causal_prob": 0.5,
    "lambda_recon": 1.0,
    "lambda_contrast": 0.5,
    "softdtw_gamma": 0.1,
    "softdtw_pair_chunk_size": 256,
    "parameter_count": parameter_count,
    "elapsed_seconds": elapsed_seconds,
    "peak_gpu_memory_mb": peak_gpu_memory_mb,
    "best_checkpoint_exists": (
        best_checkpoint.exists()
    ),
    "trainer_metrics": trainer_metrics,
}

with open(
    OUTPUT_ROOT / "metrics.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        to_jsonable(result),
        file,
        ensure_ascii=False,
        indent=2,
    )

print("\n" + "=" * 70)
print("FINAL RESULT")
print(
    json.dumps(
        to_jsonable(result),
        ensure_ascii=False,
        indent=2,
    )
)
print("Saved to:", OUTPUT_ROOT)
