"""AeroWF 评测器的文件型子进程入口；调用方不能传入数据路径。"""
import argparse
import csv
import hashlib
import json
import os
from datetime import datetime, timezone

import numpy as np


此文件目录 = os.path.dirname(os.path.abspath(__file__))
评测器根目录 = os.path.dirname(os.path.dirname(此文件目录))
必填试验字段 = {"trial_id", "arm", "seed", "task", "checkpoint_digest"}


class 提交无效(Exception):
    """调用方提交违反接口；输出 invalid。"""

    def __init__(self, 原因, 异常计数=None):
        super().__init__(原因)
        self.异常计数 = 异常计数 or {}


class 评测故障(Exception):
    """评测器、私有配置或本地数据故障；输出 failed。"""


def 初始异常计数():
    return {"nan": 0, "inf": 0, "out_of_range": 0, "missing_id": 0, "unknown_id": 0,
            "duplicate_id": 0, "shape_error": 0, "undefined_zero_denominator": 0, "zero_support": 0}


def 文件哈希(路径):
    哈希器 = hashlib.sha256()
    with open(路径, "rb") as 文件:
        for 块 in iter(lambda: 文件.read(1024 * 1024), b""):
            哈希器.update(块)
    return 哈希器.hexdigest()


def 多文件哈希(路径列表):
    """对相对路径和文件内容共同摘要，作为评测器代码版本摘要。"""
    哈希器 = hashlib.sha256()
    for 路径 in sorted(路径列表):
        相对路径 = os.path.relpath(路径, 评测器根目录).replace("\\", "/")
        哈希器.update(相对路径.encode("utf-8"))
        哈希器.update(b"\0")
        with open(路径, "rb") as 文件:
            for 块 in iter(lambda: 文件.read(1024 * 1024), b""):
                哈希器.update(块)
    return 哈希器.hexdigest()


def 写入_json(路径, 内容):
    os.makedirs(os.path.dirname(os.path.abspath(路径)), exist_ok=True)
    with open(路径, "w", encoding="utf-8") as 文件:
        json.dump(内容, 文件, ensure_ascii=False, indent=2)
        文件.write("\n")


def 读取_json(路径, 名称):
    try:
        with open(路径, encoding="utf-8") as 文件:
            return json.load(文件)
    except Exception as 异常:
        raise 评测故障(f"无法读取{名称}") from 异常


def 读取样本编号(目录):
    try:
        with open(os.path.join(目录, "index.csv"), encoding="utf-8-sig", newline="") as 文件:
            行 = list(csv.DictReader(文件))
        if not 行 or "sample_id" not in 行[0]:
            raise ValueError("index.csv 缺少 sample_id")
        编号 = np.asarray([一行["sample_id"] for 一行 in 行], dtype="U128")
        if len(np.unique(编号)) != len(编号):
            raise ValueError("私有 split 清单存在重复 sample_id")
        return 编号
    except Exception as 异常:
        raise 评测故障("无法读取私有 val 的 sample_id 清单") from 异常


def utc时间():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def 日期块编号(时间戳):
    return np.unique(时间戳.astype("datetime64[D]"), return_inverse=True)[1]


def 指标值(误差, 指标):
    if 指标 == "mae":
        return float(np.mean(np.abs(误差)))
    if 指标 == "mse":
        return float(np.mean(np.square(误差)))
    if 指标 == "rmse":
        return float(np.sqrt(np.mean(np.square(误差))))
    raise 评测故障(f"未知连续指标：{指标}")


def 连续端点(名称, 误差, 块, 指标, 配置):
    """以日期块 bootstrap；RMSE 每次均先平方平均、后开根号。"""
    if len(误差) == 0:
        return {"name": 名称, "value": None, "ci95": [None, None], "degenerate": True}
    唯一块, 块位置 = np.unique(块, return_inverse=True)
    数量 = np.bincount(块位置).astype(float)
    if 指标 == "mae":
        分子 = np.bincount(块位置, weights=np.abs(误差))
        计算 = lambda x, n: x / n
    else:
        分子 = np.bincount(块位置, weights=np.square(误差))
        计算 = (lambda x, n: x / n) if 指标 == "mse" else (lambda x, n: np.sqrt(x / n))
    值 = float(计算(float(分子.sum()), float(数量.sum())))
    if len(唯一块) < 2:
        区间 = [None, None]
    else:
        种子 = int.from_bytes(hashlib.sha256(名称.encode("utf-8")).digest()[:8], "little") ^ int(配置["bootstrap"]["seed"])
        随机数 = np.random.default_rng(种子)
        抽样 = 随机数.integers(0, len(唯一块), size=(int(配置["bootstrap"]["n_replicates"]), len(唯一块)))
        重采样值 = 计算(分子[抽样].sum(axis=1), 数量[抽样].sum(axis=1))
        下界 = (1.0 - float(配置["bootstrap"]["ci_level"])) / 2.0 * 100.0
        区间 = [float(x) for x in np.percentile(重采样值, [下界, 100.0 - 下界])]
    return {"name": 名称, "value": 值, "ci95": 区间, "degenerate": False}


def csi(真正, 假正, 假负):
    分母 = 真正 + 假正 + 假负
    return None if 分母 == 0 else float(真正 / 分母)


def f1(真正, 假正, 假负):
    分母 = 2 * 真正 + 假正 + 假负
    return None if 分母 == 0 else float(2 * 真正 / 分母)


def 分类端点(名称, 真值, 预测, 块, 类别, 配置, 异常计数):
    """固定 v2 的 zero-support 和 event-rate 退化规则。"""
    真正 = int(np.sum((真值 == 类别) & (预测 == 类别)))
    假正 = int(np.sum((真值 != 类别) & (预测 == 类别)))
    假负 = int(np.sum((真值 == 类别) & (预测 != 类别)))
    支持度 = 真正 + 假负
    基率 = float(支持度 / len(真值)) if len(真值) else None
    零支持 = 支持度 == 0
    退化 = 零支持 or 基率 in (0.0, 1.0)
    if 零支持:
        异常计数["zero_support"] += 1
    if 真正 + 假正 + 假负 == 0:
        异常计数["undefined_zero_denominator"] += 1
    if 退化:
        基本 = {"value": None, "ci95": [None, None], "support": 支持度, "fp": 假正,
                "false_positive_only": bool(零支持 and 假正 > 0), "event_rate": 基率, "degenerate": True}
        return {"csi": {"name": 名称, **基本},
                "f1": {"name": 名称.replace(".csi_", ".f1_"), **基本}}
    唯一块, 块位置 = np.unique(块, return_inverse=True)
    块混淆 = np.zeros((len(唯一块), 3), dtype=np.int64)
    np.add.at(块混淆[:, 0], 块位置, ((真值 == 类别) & (预测 == 类别)).astype(np.int64))
    np.add.at(块混淆[:, 1], 块位置, ((真值 != 类别) & (预测 == 类别)).astype(np.int64))
    np.add.at(块混淆[:, 2], 块位置, ((真值 == 类别) & (预测 != 类别)).astype(np.int64))
    if len(唯一块) < 2:
        csi区间, f1区间 = [None, None], [None, None]
    else:
        种子 = int.from_bytes(hashlib.sha256(名称.encode("utf-8")).digest()[:8], "little") ^ int(配置["bootstrap"]["seed"])
        随机数 = np.random.default_rng(种子)
        抽样 = 随机数.integers(0, len(唯一块), size=(int(配置["bootstrap"]["n_replicates"]), len(唯一块)))
        和 = 块混淆[抽样].sum(axis=1)
        csi分母 = 和[:, 0] + 和[:, 1] + 和[:, 2]
        f1分母 = 2.0 * 和[:, 0] + 和[:, 1] + 和[:, 2]
        csi值 = np.divide(和[:, 0], csi分母, out=np.full(len(和), np.nan), where=csi分母 != 0)
        f1值 = np.divide(2.0 * 和[:, 0], f1分母, out=np.full(len(和), np.nan), where=f1分母 != 0)
        下界 = (1.0 - float(配置["bootstrap"]["ci_level"])) / 2.0 * 100.0
        csi有效值, f1有效值 = csi值[np.isfinite(csi值)], f1值[np.isfinite(f1值)]
        csi区间 = [float(x) for x in np.percentile(csi有效值, [下界, 100.0 - 下界])] if len(csi有效值) else [None, None]
        f1区间 = [float(x) for x in np.percentile(f1有效值, [下界, 100.0 - 下界])] if len(f1有效值) else [None, None]
    公共 = {"support": 支持度, "fp": 假正, "false_positive_only": False,
            "event_rate": 基率, "degenerate": False}
    return {"csi": {"name": 名称, "value": csi(真正, 假正, 假负), "ci95": csi区间, **公共},
            "f1": {"name": 名称.replace(".csi_", ".f1_"), "value": f1(真正, 假正, 假负), "ci95": f1区间, **公共}}


def 分类宏端点(名称, 真值, 预测, 块, 类别列表, 指标, 配置):
    """同一次日期块重采样后再做宏平均，避免把各类别 CI 机械相加。"""
    可评类别 = []
    for 类别 in 类别列表:
        基率 = float(np.mean(真值 == 类别))
        if 基率 not in (0.0, 1.0):
            可评类别.append(类别)
    if not 可评类别:
        return {"name": 名称, "value": None, "ci95": [None, None], "degenerate": True, "included_endpoint_count": 0}
    def 单类值(真正, 假正, 假负):
        return f1(真正, 假正, 假负) if 指标 == "f1" else csi(真正, 假正, 假负)
    完整值 = []
    for 类别 in 可评类别:
        完整值.append(单类值(int(np.sum((真值 == 类别) & (预测 == 类别))),
                         int(np.sum((真值 != 类别) & (预测 == 类别))),
                         int(np.sum((真值 == 类别) & (预测 != 类别)))))
    唯一块, 块位置 = np.unique(块, return_inverse=True)
    if len(唯一块) < 2:
        区间 = [None, None]
    else:
        块混淆 = np.zeros((len(可评类别), len(唯一块), 3), dtype=np.int64)
        for 位置, 类别 in enumerate(可评类别):
            np.add.at(块混淆[位置, :, 0], 块位置, ((真值 == 类别) & (预测 == 类别)).astype(np.int64))
            np.add.at(块混淆[位置, :, 1], 块位置, ((真值 != 类别) & (预测 == 类别)).astype(np.int64))
            np.add.at(块混淆[位置, :, 2], 块位置, ((真值 == 类别) & (预测 != 类别)).astype(np.int64))
        种子 = int.from_bytes(hashlib.sha256(名称.encode("utf-8")).digest()[:8], "little") ^ int(配置["bootstrap"]["seed"])
        抽样 = np.random.default_rng(种子).integers(0, len(唯一块), size=(int(配置["bootstrap"]["n_replicates"]), len(唯一块)))
        和 = 块混淆[:, 抽样, :].sum(axis=2)
        if 指标 == "f1":
            分子, 分母 = 2.0 * 和[:, :, 0], 2.0 * 和[:, :, 0] + 和[:, :, 1] + 和[:, :, 2]
        else:
            分子, 分母 = 和[:, :, 0], 和[:, :, 0] + 和[:, :, 1] + 和[:, :, 2]
        值 = np.divide(分子, 分母, out=np.full(分子.shape, np.nan), where=分母 != 0)
        宏值 = np.nanmean(值, axis=0)
        宏值 = 宏值[np.isfinite(宏值)]
        下界 = (1.0 - float(配置["bootstrap"]["ci_level"])) / 2.0 * 100.0
        区间 = [float(x) for x in np.percentile(宏值, [下界, 100.0 - 下界])] if len(宏值) else [None, None]
    return {"name": 名称, "value": float(np.mean(完整值)), "ci95": 区间,
            "degenerate": False, "included_endpoint_count": len(可评类别)}


def 有置信区间(端点):
    return "ci95" in 端点 and None not in 端点["ci95"]


def 跨端点等权平均(名称, 端点列表):
    可评估 = [端点 for 端点 in 端点列表 if not 端点.get("degenerate", False) and 端点.get("value") is not None]
    if not 可评估:
        return {"name": 名称, "value": None, "ci95": [None, None], "degenerate": True, "included_endpoint_count": 0}
    值 = float(np.mean([端点["value"] for 端点 in 可评估]))
    有区间 = [端点 for 端点 in 可评估 if 有置信区间(端点)]
    区间 = [float(np.mean([端点["ci95"][0] for 端点 in 有区间])), float(np.mean([端点["ci95"][1] for 端点 in 有区间]))] if 有区间 else [None, None]
    return {"name": 名称, "value": 值, "ci95": 区间, "degenerate": False, "included_endpoint_count": len(可评估)}


def 读取提交(参数, 配置):
    计数 = 初始异常计数()
    元信息 = 读取_json(参数.trial_meta, "trial_meta.json")
    缺少 = 必填试验字段 - set(元信息)
    if 缺少:
        raise 提交无效("trial_meta 缺少 " + ", ".join(sorted(缺少)), 计数)
    任务 = 元信息["task"]
    if 任务 not in ("forecast", "classification", "imputation"):
        raise 提交无效("未知 task", 计数)
    try:
        with np.load(参数.predictions, allow_pickle=False) as 压缩包:
            if set(压缩包.files) != {"sample_id", "pred"}:
                raise 提交无效("predictions.npz 必须恰有 sample_id 和 pred", 计数)
            样本编号 = np.asarray(压缩包["sample_id"]).astype("U128")
            预测 = np.asarray(压缩包["pred"])
    except 提交无效:
        raise
    except Exception as 异常:
        raise 提交无效("无法安全读取 predictions.npz", 计数) from 异常
    if len(样本编号) == 0:
        raise 提交无效("sample_id 为空", 计数)
    重复数 = int(len(样本编号) - len(np.unique(样本编号)))
    if 重复数:
        计数["duplicate_id"] = 重复数
        raise 提交无效("sample_id 重复", 计数)
    期望尾形状 = {"forecast": (4, 3, 2), "classification": (), "imputation": (4, 4, 96, 11)}[任务]
    if tuple(预测.shape[1:]) != 期望尾形状 or 预测.shape[0] != len(样本编号):
        计数["shape_error"] = 1
        raise 提交无效("pred 形状错误", 计数)
    if 任务 in ("forecast", "imputation"):
        if 预测.dtype.kind != "f":
            计数["shape_error"] = 1
            raise 提交无效("连续预测必须是浮点数组", 计数)
        计数["nan"] = int(np.isnan(预测).sum())
        计数["inf"] = int(np.isinf(预测).sum())
        if 计数["nan"] or 计数["inf"]:
            raise 提交无效("预测含 NaN 或 Inf", 计数)
        下限, 上限 = 配置["forecast"]["valid_range"]
        计数["out_of_range"] = int(((预测 < 下限) | (预测 > 上限)).sum())
        if 计数["out_of_range"]:
            raise 提交无效("连续预测越界，禁止 clip", 计数)
    else:
        if 预测.dtype.kind not in "iu":
            计数["shape_error"] = 1
            raise 提交无效("分类预测必须是整数数组", 计数)
        计数["out_of_range"] = int(((预测 < 0) | (预测 > 2)).sum())
        if 计数["out_of_range"]:
            raise 提交无效("分类 id 越界，禁止修正", 计数)
    return 元信息, 任务, 样本编号, 预测, 计数


def 校验完整样本编号(提交编号, 私有编号, 计数):
    提交集合, 私有集合 = set(提交编号.tolist()), set(私有编号.tolist())
    计数["missing_id"] = len(私有集合 - 提交集合)
    计数["unknown_id"] = len(提交集合 - 私有集合)
    if 计数["missing_id"] or 计数["unknown_id"]:
        raise 提交无效("sample_id 与完整 val 清单不一致", 计数)


def 加载机场数据(私有配置, 配置):
    数据, 所有编号 = {}, []
    try:
        for 机场 in 配置["airports"]:
            目录 = os.path.join(私有配置["val_data_root"], "trainval", "val", 机场)
            样本编号 = 读取样本编号(目录)
            数据[机场] = {"目录": 目录, "sample_id": 样本编号,
                         "runway": np.load(os.path.join(目录, "runway.npy"), allow_pickle=False),
                         "runway_mask": np.load(os.path.join(目录, "runway_mask.npy"), allow_pickle=False).astype(bool),
                         "timestamps": np.load(os.path.join(目录, "timestamps.npy"), allow_pickle=False),
                         "weather_label": np.load(os.path.join(目录, "weather_label.npy"), allow_pickle=False)}
            所有编号.extend(样本编号.tolist())
    except 评测故障:
        raise
    except Exception as 异常:
        raise 评测故障("无法读取 C 私有 val 真值") from 异常
    return 数据, np.asarray(所有编号, dtype="U128")


def 评测预测(预测, 数据, 配置, 异常计数):
    端点 = []
    for 机场, 机场数据 in 数据.items():
        跑道真值, 跑道掩码, 时间戳 = 机场数据["runway"], 机场数据["runway_mask"], 机场数据["timestamps"]
        块 = 日期块编号(时间戳)
        越界 = (预测[机场] < 配置["forecast"]["valid_range"][0]) | (预测[机场] > 配置["forecast"]["valid_range"][1])
        if 越界.any():
            异常计数["out_of_range"] += int(越界.sum())
            raise 提交无效("预测值越界，禁止 clip", 异常计数)
        分钟 = 时间戳.astype("datetime64[m]").astype("int64")
        位置查询 = {int(值): 位置 for 位置, 值 in enumerate(分钟)}
        for 预测位置, (跨度名, 分钟偏移) in enumerate(配置["forecast"]["horizons"].items()):
            伙伴位置 = np.asarray([位置查询.get(int(值 + 分钟偏移), -1) for 值 in 分钟])
            有效 = 跑道掩码 & (伙伴位置 >= 0)[:, None]
            行, 跑道槽 = np.where(有效)
            for 分量位置, 分量名 in enumerate(配置["forecast"]["components"]):
                误差 = 预测[机场][行, 跑道槽, 预测位置, 分量位置] - 跑道真值[伙伴位置[行], 跑道槽, 95, 分量位置 + 1]
                前缀 = f"forecast.{机场}.{跨度名}.{分量名}"
                端点.append(连续端点(f"{前缀}.mae", 误差, 块[行], "mae", 配置))
                端点.append(连续端点(f"{前缀}.rmse", 误差, 块[行], "rmse", 配置))
    总体 = [跨端点等权平均("overall.forecast.mae", [端点 for 端点 in 端点 if 端点["name"].endswith(".mae")]),
            跨端点等权平均("overall.forecast.rmse", [端点 for 端点 in 端点 if 端点["name"].endswith(".rmse")])]
    return 端点, 总体


def 评测分类(预测, 数据, 标签组, 配置, 异常计数):
    端点 = []
    for 机场, 机场数据 in 数据.items():
        映射 = np.asarray(标签组["upstream_to_task_class"])
        原始标签 = 机场数据["weather_label"]
        if np.any((原始标签 < 0) | (原始标签 >= len(映射))):
            raise 评测故障("私有 val 的 weather_label 超出合同映射范围")
        真值, 有效 = 映射[原始标签], 映射[原始标签] >= 0
        真值, 机场预测 = 真值[有效], 预测[机场][有效]
        块 = 日期块编号(机场数据["timestamps"])[有效]
        分类结果 = []
        for 类别名, 类别号 in zip(标签组["class_names"], [0, 1, 2]):
            结果 = 分类端点(f"classification.{机场}.csi_{类别名.lower()}", 真值, 机场预测, 块, 类别号, 配置, 异常计数)
            端点.append(结果["csi"])
            分类结果.append(结果)
        端点.append(分类宏端点(f"classification.{机场}.macro_f1", 真值, 机场预测, 块, [0, 1, 2], "f1", 配置))
        端点.append(分类宏端点(f"classification.{机场}.csi_macro", 真值, 机场预测, 块, [0, 1, 2], "csi", 配置))
    总体 = [跨端点等权平均("overall.classification.macro_f1", [端点 for 端点 in 端点 if 端点["name"].endswith(".macro_f1")]),
            跨端点等权平均("overall.classification.csi_macro", [端点 for 端点 in 端点 if 端点["name"].endswith(".csi_macro")]),
            跨端点等权平均("overall.classification.csi_good", [端点 for 端点 in 端点 if 端点["name"].endswith(".csi_good")]),
            跨端点等权平均("overall.classification.csi_precip", [端点 for 端点 in 端点 if 端点["name"].endswith(".csi_precip")]),
            跨端点等权平均("overall.classification.csi_class_hazard", [端点 for 端点 in 端点 if 端点["name"].endswith(".csi_class_hazard")])]
    return 端点, 总体


def 评测插补(预测, 数据, 元信息, 私有配置, 配置):
    端点 = []
    私有掩码清单 = 读取_json(os.path.join(私有配置["mask_root"], "MANIFEST.json"), "C 私有固定 mask 清单")
    冻结清单路径 = os.path.join(评测器根目录, "config", "fixed_mask_manifest_val_v1.json")
    if 文件哈希(冻结清单路径) != 配置["imputation"]["mask_manifest_sha256"]:
        raise 评测故障("冻结 mask 清单与 evaluator_config 不一致")
    冻结清单 = 读取_json(冻结清单路径, "冻结 mask 清单")
    for 机场, 机场数据 in 数据.items():
        for 情景位置, 情景 in enumerate(配置["imputation"]["scenarios"]):
            掩码路径 = os.path.join(私有配置["mask_root"], "validation", 机场, f"imputation_{情景}.npz")
            try:
                实际哈希 = 文件哈希(掩码路径)
                私有哈希 = 私有掩码清单["masks"]["validation"][机场][情景]
                冻结哈希 = 冻结清单["masks"]["validation"][机场][情景]
                if 实际哈希 != 私有哈希 or 实际哈希 != 冻结哈希:
                    raise ValueError("固定 mask 哈希不匹配")
                掩码 = np.load(掩码路径, allow_pickle=False)["mask"].astype(bool)
            except Exception as 异常:
                raise 评测故障("固定 mask 校验失败") from 异常
            有效掩码 = 掩码 & 机场数据["runway_mask"][:, :, None, None]
            误差 = 预测[机场][:, 情景位置] - 机场数据["runway"]
            每样本数量 = 有效掩码.reshape(有效掩码.shape[0], -1).sum(axis=1)
            块 = np.repeat(日期块编号(机场数据["timestamps"]), 每样本数量)
            前缀 = f"imputation.{机场}.{情景}"
            端点.append(连续端点(f"{前缀}.mse", 误差[有效掩码], 块, "mse", 配置))
            端点.append(连续端点(f"{前缀}.mae", 误差[有效掩码], 块, "mae", 配置))
    总体 = [跨端点等权平均("overall.imputation.mse", [端点 for 端点 in 端点 if 端点["name"].endswith(".mse")]),
            跨端点等权平均("overall.imputation.mae", [端点 for 端点 in 端点 if 端点["name"].endswith(".mae")])]
    return 端点, 总体


def 构造清单(参数, 私有配置, 配置, 样本数):
    代码文件 = [os.path.join(此文件目录, "__init__.py"), os.path.join(此文件目录, "__main__.py")]
    return {"utc": utc时间(), "evaluator_version": 配置["evaluator_version"],
            "evaluator_code_sha256": 多文件哈希(代码文件),
            "evaluator_config_sha256": 文件哈希(os.path.join(评测器根目录, "config", "evaluator_config_v1.json")),
            "split_manifest_sha256": 文件哈希(私有配置["split_manifest"]),
            "normalization_stats_sha256": 文件哈希(私有配置["normalization_stats"]),
            "metric_definition_id": 配置["metric_definition_id"],
            "task_contract_version": 配置["task_contract_version"], "task_contract_sha256": 配置["task_contract_sha256"],
            "predictions_sha256": 文件哈希(参数.predictions), "trial_meta_sha256": 文件哈希(参数.trial_meta),
            "n_samples": 样本数}


def 运行(参数):
    if 参数.split != "val":
        raise 提交无效("--split 只允许 val")
    私有配置路径 = os.environ.get("AEROWF_EVALUATOR_PRIVATE_CONFIG")
    if not 私有配置路径:
        raise 评测故障("未安装 C 私有配置")
    私有配置 = 读取_json(私有配置路径, "C 私有配置")
    配置 = 读取_json(os.path.join(评测器根目录, "config", "evaluator_config_v1.json"), "冻结评测配置")
    标签组 = 读取_json(os.path.join(评测器根目录, "config", "label_groups_v1.json"), "冻结分类标签组")
    元信息, 任务, 提交编号, 原始预测, 异常计数 = 读取提交(参数, 配置)
    数据, 私有编号 = 加载机场数据(私有配置, 配置)
    校验完整样本编号(提交编号, 私有编号, 异常计数)
    行号 = {编号: 位置 for 位置, 编号 in enumerate(提交编号.tolist())}
    按机场预测 = {机场: 原始预测[np.asarray([行号[编号] for 编号 in 内容["sample_id"]])] for 机场, 内容 in 数据.items()}
    if 任务 == "forecast":
        端点, 总体 = 评测预测(按机场预测, 数据, 配置, 异常计数)
    elif 任务 == "classification":
        端点, 总体 = 评测分类(按机场预测, 数据, 标签组, 配置, 异常计数)
    else:
        端点, 总体 = 评测插补(按机场预测, 数据, 元信息, 私有配置, 配置)
    指标 = {"status": "completed", "task": 任务, "split": "val", "task_contract_version": 配置["task_contract_version"],
            "contract_sha256": 配置["task_contract_sha256"],
            "trial": {键: 元信息[键] for 键 in sorted(必填试验字段)}, "endpoints": 端点, "overall": 总体,
            "anomaly_counts": 异常计数}
    return 指标, 构造清单(参数, 私有配置, 配置, len(私有编号))


def main():
    解析器 = argparse.ArgumentParser()
    解析器.add_argument("--predictions", required=True)
    解析器.add_argument("--trial-meta", required=True)
    解析器.add_argument("--split", required=True)
    解析器.add_argument("--out-dir", required=True)
    参数 = 解析器.parse_args()
    try:
        指标, 清单 = 运行(参数)
        写入_json(os.path.join(参数.out_dir, "metrics.json"), 指标)
        写入_json(os.path.join(参数.out_dir, "evaluation_manifest.json"), 清单)
        return 0
    except 提交无效 as 异常:
        计数 = 初始异常计数()
        计数.update(异常.异常计数)
        写入_json(os.path.join(参数.out_dir, "metrics.json"), {"status": "invalid", "not_evaluated": True,
                 "reason": str(异常), "anomaly_counts": 计数})
        return 2
    except Exception as 异常:
        写入_json(os.path.join(参数.out_dir, "metrics.json"), {"status": "failed", "not_evaluated": True,
                 "reason": str(异常), "anomaly_counts": 初始异常计数()})
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
