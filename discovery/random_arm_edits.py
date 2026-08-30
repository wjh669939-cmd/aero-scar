"""随机对照臂：模板 + 参数 → 机械生成编辑文件（人类预实现，零 LLM）。

采样空间 random_space_v1（2026-08-30 预注册）：
    O1-horizon-weighted-mse / O2-event-weighted-loss / O3-classwise-focal /
    R4-runway-frame-wind / R5-wind-cyclic-encoding / R6-multiscale-diff
排除并留痕：
    R1（code_level 未预实现）；R2/R3（实现依赖尚未核实的通道语义，核实后补入）；
    tier2 三模板（2.8h/发，本期两臂对照先覆盖 R/O-t1 空间）。
两臂对照（LLM vs random）在本空间子集内配对比较，论文如实注明。

事实依据（均已实测验证）：
    - runway_arr 布局 (n_slots, 96, 11)；wind_x=ch1, wind_y=ch2（冻结靶通道）；
      注：G-13 演练的 R5 参考实现变换的是 ch0/ch1，与靶通道不符，本库不沿用；
    - runway_axis_heading_deg: (n_slots,) mod-180 名义朝向，NaN=虚拟/未知槽位；
    - forecast_loss prediction/target 布局 (batch, slots, horizons=3, comps=2)；
    - event_mask/event_available: (batch, 3) bool，train-only。
"""

from __future__ import annotations

import json
from pathlib import Path

TRIAL_FEATURES_BASELINE = Path("/root/autodl-tmp/aerowf_downstream_v2/src/trial_features.py")
TRIAL_OBJECTIVE_BASELINE = Path("/root/autodl-tmp/aerowf_downstream_v2/src/trial_objective.py")

_X_LINE = '"x": torch.from_numpy(np.array(runway_arr, dtype=np.float32, copy=True)),'

RANDOM_SPACE_V1 = (
    "O1-horizon-weighted-mse",
    "O2-event-weighted-loss",
    "O3-classwise-focal",
    "R4-runway-frame-wind",
    "R5-wind-cyclic-encoding",
    "R6-multiscale-diff",
)


def _patch_features(helper: str, x_expr: str, uses_heading: bool = False) -> str:
    src = TRIAL_FEATURES_BASELINE.read_text(encoding="utf-8")
    anchor = "def build_forecast_inputs("
    assert anchor in src and src.count(_X_LINE) == 2, "baseline drifted; regenerate library"
    src = src.replace(anchor, helper + "\n\n" + anchor, 1)
    if uses_heading:
        # 基线用 del 丢弃朝向参数；消费方必须把 del 摘掉
        src = src.replace(
            "    del runway_axis_heading_deg  # baseline ignores runway geometry\n", ""
        )
    return src.replace(_X_LINE, f'"x": {x_expr},')


def _r4(params: dict) -> str:
    components = list(params["components"])
    helper = f'''_R4_COMPONENTS = {components!r}


def _runway_frame(runway_arr: np.ndarray, heading_deg) -> np.ndarray:
    """R4: 把 wind_x/wind_y(ch1/ch2) 旋转到跑道轴坐标系（顺风/侧风分量）。

    heading 为 mod-180 名义朝向（度）；NaN 槽位保持原始分量（容忍语义）。
    分量选择: headwind -> ch1, crosswind -> ch2；只选其一时另一通道保留原值。
    gust_crosswind 依赖阵风矢量数据（接口未提供），按注册参数如实实现为
    crosswind 同值（阵风标量不可分解），并在此注明近似。
    """
    arr = np.array(runway_arr, dtype=np.float32, copy=True)
    if heading_deg is None:
        return arr
    theta = np.deg2rad(np.asarray(heading_deg, dtype=np.float32))  # (n_slots,)
    valid = np.isfinite(theta)
    if not valid.any():
        return arr
    u = arr[..., 1] - 0.5
    v = arr[..., 2] - 0.5
    sin_t = np.sin(theta)[:, None]
    cos_t = np.cos(theta)[:, None]
    headwind = u * sin_t + v * cos_t
    crosswind = -u * cos_t + v * sin_t
    vmask = valid[:, None]
    if "headwind" in _R4_COMPONENTS:
        arr[..., 1] = np.where(vmask, np.clip(headwind + 0.5, 0.0, 1.0), arr[..., 1])
    if "crosswind" in _R4_COMPONENTS or "gust_crosswind" in _R4_COMPONENTS:
        arr[..., 2] = np.where(vmask, np.clip(crosswind + 0.5, 0.0, 1.0), arr[..., 2])
    return arr'''
    return _patch_features(
        helper, "torch.from_numpy(_runway_frame(runway_arr, runway_axis_heading_deg))",
        uses_heading=True,
    )


def _r5(params: dict) -> str:
    variant = params["variant"]
    helper = f'''_R5_VARIANT = {variant!r}


def _wind_cyclic(runway_arr: np.ndarray) -> np.ndarray:
    """R5: 风向循环编码，作用于冻结靶通道 wind_x=ch1 / wind_y=ch2。

    sincos_only: (ch1,ch2) <- 方向 (sin,cos)（丢幅值）；
    sincos_plus_speed_decouple: ch1 <- 归一化幅值 r，ch2 <- (sinθ+1)/2（幅值-方向解耦）。
    """
    arr = np.array(runway_arr, dtype=np.float32, copy=True)
    u = arr[..., 1] - 0.5
    v = arr[..., 2] - 0.5
    theta = np.arctan2(v, u)
    if _R5_VARIANT == "sincos_only":
        arr[..., 1] = (np.sin(theta) + 1.0) / 2.0
        arr[..., 2] = (np.cos(theta) + 1.0) / 2.0
    else:
        r = np.sqrt(u * u + v * v)
        arr[..., 1] = np.clip(r * np.sqrt(2.0), 0.0, 1.0)
        arr[..., 2] = (np.sin(theta) + 1.0) / 2.0
    return arr'''
    return _patch_features(helper, "torch.from_numpy(_wind_cyclic(runway_arr))")


def _r6(params: dict) -> str:
    steps = list(params["diff_steps"])
    window = params["rolling_var_window"]
    helper = f'''_R6_STEPS = {steps!r}
_R6_WINDOW = {window!r}


def _multiscale_diff(runway_arr: np.ndarray) -> np.ndarray:
    """R6: 多尺度差分按注册语义实现为追加通道（结构可行性由闸门裁决）。"""
    arr = np.array(runway_arr, dtype=np.float32, copy=True)
    extras = []
    for step in _R6_STEPS:
        d = np.zeros_like(arr[..., 1:3])
        d[:, step:, :] = arr[:, step:, 1:3] - arr[:, :-step, 1:3]
        extras.append(d)
    if _R6_WINDOW:
        w = _R6_WINDOW
        var = np.zeros_like(arr[..., 1:3])
        for t in range(arr.shape[1]):
            lo = max(0, t - w + 1)
            var[:, t, :] = arr[:, lo:t + 1, 1:3].var(axis=1)
        extras.append(var)
    return np.concatenate([arr] + extras, axis=-1)'''
    return _patch_features(helper, "torch.from_numpy(_multiscale_diff(runway_arr))")


def _o1(params: dict) -> str:
    weights = list(params["weights_t1_t4_t8"])
    src = TRIAL_OBJECTIVE_BASELINE.read_text(encoding="utf-8")
    old = """    mask = node_mask[:, :, None, None].expand_as(prediction)
    return torch.square(prediction - target)[mask].mean()"""
    new = f"""    mask = node_mask[:, :, None, None].expand_as(prediction)
    weights = torch.tensor({weights!r}, device=prediction.device, dtype=prediction.dtype)
    weighted = torch.square(prediction - target) * weights.view(1, 1, -1, 1)
    return weighted[mask].mean()"""
    assert old in src
    return src.replace(old, new)


def _o2(params: dict) -> str:
    w = params["event_weight"]
    src = TRIAL_OBJECTIVE_BASELINE.read_text(encoding="utf-8")
    old = """    mask = node_mask[:, :, None, None].expand_as(prediction)
    return torch.square(prediction - target)[mask].mean()"""
    new = f"""    mask = node_mask[:, :, None, None].expand_as(prediction)
    sq = torch.square(prediction - target)
    if event_mask is not None:
        ev = event_mask
        if event_available is not None:
            ev = ev & event_available  # 不可用时距不作为事件加权对象
        w = 1.0 + ({w} - 1.0) * ev.to(sq.dtype)  # (batch, horizons)
        sq = sq * w[:, None, :, None]
    return sq[mask].mean()"""
    assert old in src
    return src.replace(old, new)


def _o3(params: dict) -> str:
    scheme, gamma = params["alpha_scheme"], params["gamma"]
    src = TRIAL_OBJECTIVE_BASELINE.read_text(encoding="utf-8")
    old = """    return F.cross_entropy(
        logits, label, weight=class_weights, ignore_index=-100
    )"""
    new = f"""    # O3 classwise focal: alpha={scheme!r}, gamma={gamma}
    alpha = class_weights if {scheme!r} == "inverse_freq" else torch.sqrt(class_weights)
    ce = F.cross_entropy(logits, label, weight=alpha.to(logits.device),
                         ignore_index=-100, reduction="none")
    valid = label != -100
    pt = torch.exp(-ce)
    focal = ((1.0 - pt) ** {gamma}) * ce
    return focal[valid].mean() if valid.any() else focal.sum() * 0.0"""
    assert old in src
    return src.replace(old, new)


_GENERATORS = {
    "R4-runway-frame-wind": ("representation", _r4),
    "R5-wind-cyclic-encoding": ("representation", _r5),
    "R6-multiscale-diff": ("representation", _r6),
    "O1-horizon-weighted-mse": ("objective_tier1", _o1),
    "O2-event-weighted-loss": ("objective_tier1", _o2),
    "O3-classwise-focal": ("objective_tier1", _o3),
}


def generate_edit(action_id: str, params: dict) -> tuple[str, str]:
    """返回 (axis, 完整编辑文件内容)。"""
    axis, fn = _GENERATORS[action_id]
    return axis, fn(params)


def sample_params(action, rng) -> dict:
    out = {}
    for name, spec in (action.get("param_space") or {}).items():
        kind = spec["kind"]
        if kind == "choice":
            out[name] = rng.choice(spec["choices"])
        elif kind == "subset_nonempty":
            choices = list(spec["choices"])
            k = rng.randint(1, len(choices))
            out[name] = sorted(rng.sample(choices, k), key=str)
        else:
            raise ValueError(f"unknown param kind {kind}")
    return out
