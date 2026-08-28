"""M-axis validators (ENGINEERING-TIER stub; DEC-001 activation prerequisite).

三道检查（激活时补全 IO/前后向的真实构造，参照 tier2 冒烟的缩维配置）：
1. param_budget: |params(candidate) - params(parent)| / params(parent) <= 0.05
2. io_shapes: encode/forward 输出形状与 parent 一致
3. fwd_bwd_finite: 缩维假张量前后向损失与梯度有限
"""

PARENT_PARAMETER_COUNT = None  # DEC-001 激活时以 parent checkpoint 实测值冻结


def param_budget(candidate_param_count: int, parent_param_count: int, tol: float = 0.05) -> str | None:
    if parent_param_count <= 0:
        return "parent_param_count invalid"
    rel = abs(candidate_param_count - parent_param_count) / parent_param_count
    if rel > tol:
        return f"param budget exceeded: |delta|={rel:.4f} > {tol}"
    return None


def io_shapes(*_args, **_kwargs):
    raise NotImplementedError("implement at DEC-001 activation (2026-09-02 review)")


def fwd_bwd_finite(*_args, **_kwargs):
    raise NotImplementedError("implement at DEC-001 activation (2026-09-02 review)")
