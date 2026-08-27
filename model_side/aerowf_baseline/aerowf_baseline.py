from pathlib import Path
import json
import time
import subprocess

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

DATA_ROOT = Path("data/AeroWF/processed")
RESULT_ROOT = Path("results/aerowf_logreg_v1")
RESULT_ROOT.mkdir(parents=True, exist_ok=True)

AIRPORTS = ["ZBAA", "ZBAD", "ZSPD", "ZSSS"]
LABELS = [2, 9, 16, 17]
LABEL_NAMES = {
    2: "GOOD_WX",
    9: "RA",
    16: "BR",
    17: "HZ",
}
PURGE_WINDOWS = 95


def extract_features(x):
    """
    x: [N, R, T, F]

    每个特征先在跑道维度取平均，再提取：
    1. 最后时刻
    2. 24小时均值
    3. 标准差
    4. 最小值
    5. 最大值
    6. 最后8帧与最初8帧的均值差
    """
    runway_mean = x.mean(axis=1)

    last = runway_mean[:, -1, :]
    mean = runway_mean.mean(axis=1)
    std = runway_mean.std(axis=1)
    minimum = runway_mean.min(axis=1)
    maximum = runway_mean.max(axis=1)
    trend = (
        runway_mean[:, -8:, :].mean(axis=1)
        - runway_mean[:, :8, :].mean(axis=1)
    )

    return np.concatenate(
        [last, mean, std, minimum, maximum, trend],
        axis=1,
    ).astype(np.float32)


def add_airport_one_hot(x, airport_index):
    one_hot = np.zeros((len(x), len(AIRPORTS)), dtype=np.float32)
    one_hot[:, airport_index] = 1.0
    return np.concatenate([x, one_hot], axis=1)

def calculate_metrics(y_true, y_pred):
    true_present_labels = np.unique(y_true)

    return {
        "accuracy": float(
            accuracy_score(y_true, y_pred)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, y_pred)
        ),
        "macro_f1_true_present_classes": float(
            f1_score(
                y_true,
                y_pred,
                labels=true_present_labels,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_f1_fixed_4_classes": float(
            f1_score(
                y_true,
                y_pred,
                labels=LABELS,
                average="macro",
                zero_division=0,
            )
        ),
    }

train_features = []
train_labels = []
val_features = []
val_labels = []
val_airports = []

print("Loading and extracting features...")

for airport_index, airport in enumerate(AIRPORTS):
    path = DATA_ROOT / f"{airport}_train.npy"
    data = np.load(path, allow_pickle=True).item()

    x_train = extract_features(data["train_runway"])
    y_train = data["train_weather_label"].astype(np.int64)

    # 剔除验证集最前面的95个窗口，减少跨边界窗口重叠
    x_val = extract_features(data["val_runway"][PURGE_WINDOWS:])
    y_val = data["val_weather_label"][PURGE_WINDOWS:].astype(np.int64)

    x_train = add_airport_one_hot(x_train, airport_index)
    x_val = add_airport_one_hot(x_val, airport_index)

    train_features.append(x_train)
    train_labels.append(y_train)
    val_features.append(x_val)
    val_labels.append(y_val)
    val_airports.extend([airport] * len(y_val))

    print(
        airport,
        "train =", x_train.shape,
        "val =", x_val.shape,
        "train labels =", dict(zip(*np.unique(y_train, return_counts=True))),
        "val labels =", dict(zip(*np.unique(y_val, return_counts=True))),
    )

    del data, x_train, x_val


X_train = np.concatenate(train_features)
y_train = np.concatenate(train_labels)
X_val = np.concatenate(val_features)
y_val = np.concatenate(val_labels)
val_airports = np.asarray(val_airports)

print("\nCombined train:", X_train.shape)
print("Combined val:", X_val.shape)

# Baseline 1：始终预测训练集多数类
majority_class = int(
    np.bincount(y_train, minlength=21).argmax()
)
majority_pred = np.full_like(y_val, majority_class)

results = {
    "task": "current weather classification from runway sequence",
    "test_used": False,
    "validation_purge_windows": PURGE_WINDOWS,
    "input_features": (
        "runway-only: last/mean/std/min/max/trend + airport one-hot"
    ),
    "labels": LABEL_NAMES,
    "majority_class": majority_class,
    "majority": calculate_metrics(y_val, majority_pred),
}

print("\nMajority baseline:")
print(json.dumps(results["majority"], indent=2))

# Baseline 2：类别平衡的多分类逻辑回归
model = Pipeline([
    ("scaler", StandardScaler()),
    (
        "classifier",
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=42,
        ),
    ),
])

print("\nTraining Logistic Regression...")
start_time = time.time()
model.fit(X_train, y_train)
training_seconds = time.time() - start_time

val_pred = model.predict(X_val)
results["logistic_regression"] = calculate_metrics(y_val, val_pred)
results["training_seconds"] = training_seconds

print("\nLogistic Regression:")
print(json.dumps(results["logistic_regression"], indent=2))
print("Training seconds:", training_seconds)

report = classification_report(
    y_val,
    val_pred,
    labels=LABELS,
    target_names=[LABEL_NAMES[i] for i in LABELS],
    zero_division=0,
    output_dict=True,
)

matrix = confusion_matrix(y_val, val_pred, labels=LABELS)

results["per_airport"] = {}

for airport in AIRPORTS:
    mask = val_airports == airport

    results["per_airport"][airport] = {
        "n_samples": int(mask.sum()),
        "majority": calculate_metrics(
            y_val[mask], majority_pred[mask]
        ),
        "logistic_regression": calculate_metrics(
            y_val[mask], val_pred[mask]
        ),
        "true_distribution": {
            str(int(k)): int(v)
            for k, v in zip(*np.unique(y_val[mask], return_counts=True))
        },
        "predicted_distribution": {
            str(int(k)): int(v)
            for k, v in zip(*np.unique(val_pred[mask], return_counts=True))
        },
    }

try:
    git_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        text=True,
    ).strip()
except Exception:
    git_commit = "unknown"

results["git_commit"] = git_commit

with open(RESULT_ROOT / "metrics.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

with open(
    RESULT_ROOT / "classification_report.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(report, f, indent=2, ensure_ascii=False)

np.savetxt(
    RESULT_ROOT / "confusion_matrix.csv",
    matrix,
    delimiter=",",
    fmt="%d",
    header="GOOD_WX,RA,BR,HZ",
    comments="",
)

np.savez_compressed(
    RESULT_ROOT / "validation_predictions.npz",
    y_true=y_val,
    y_pred=val_pred,
    airport=val_airports,
)

print("\nSaved to:", RESULT_ROOT)
print("Git commit:", git_commit)
print("\nPer-airport results:")
print(json.dumps(results["per_airport"], indent=2, ensure_ascii=False))
