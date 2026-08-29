"""AeroWF-v1 正式 discovery 驱动（LLM 臂，筛选期）。

每个 trial 的闭环：
  1. context_assembler 组装提案提示词（隐藏 token 检查）→ DeepSeek 出提案 JSON；
  2. proposal_parser 严格解析 → trial 记录（过冻结 schema）；
  3. 第二段 LLM：当前轴文件全文 + 接口合同 + patch_plan → 新文件全文；
     py_compile + 接口签名存在性 + 隐藏 token + 禁 import 检查；
  4. axis_lock 校验 → 备份并安装编辑；
  5. 正式 full_pipeline（100 预训练 + 30 下游 epoch，seed 42，六条件判定）；
  6. 四条下游腿全部过适配器 + C evaluator（v1.0.2）；
  7. result.json 过冻结 schema；lineage.jsonl 追加（配对 Δ vs parent seed42）；
  8. 还原轴文件（finally 保证）。

失败的 trial 完整落盘并写入 lineage（假设/失败原因可被下一轮提案看到），不进入候选比较。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import py_compile
import re
import shutil
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, "/root/autodl-tmp/clh_deploy/workspace/runs/aerowf-v1/tools")

from contract_tools.axis_lock import check, load_config
from contract_tools.context_assembler import (
    INTERFACE_CONTRACTS,
    assemble_proposal_prompt,
    assert_no_hidden_tokens,
)
from contract_tools.evidence_builder import build_failure_slices
from contract_tools.free_proposal import FORCED_FREE_DIRECTIVE, forced_free_status
from contract_tools.evaluator_client import run_evaluator
from contract_tools.pipeline_executor import PipelineConfig, run_pipeline
from contract_tools.predictions_adapter import adapt_stage_npz
from contract_tools.proposal_parser import parse_llm_proposal
from contract_tools.validate import validate_result

CONTRACT = Path("/root/autodl-tmp/clh_deploy/workspace/runs/aerowf-v1/00_contract")
REGISTRY = CONTRACT / "action_registry_v1_draft.json"
AEROWF_REPO = "/root/autodl-tmp/aerowf_baseline/AeroWF"
DOWNSTREAM = "/root/autodl-tmp/aerowf_downstream_v2"
DISCOVERY = Path("/root/autodl-tmp/clh_deploy/discovery")
LINEAGE = DISCOVERY / "lineage.jsonl"
COUNTS = {"ZBAA": 3961, "ZSPD": 3961, "ZSSS": 3961}
EVAL_CMD = ["/root/miniconda3/bin/python", "-m", "evaluator.aerowf_evaluator"]

AXIS_FILES = {
    "representation": Path(DOWNSTREAM) / "src/trial_features.py",
    "objective_tier1": Path(DOWNSTREAM) / "src/trial_objective.py",
    "objective_tier2": Path(AEROWF_REPO) / "models/AirFM/unified_model.py",
}
REQUIRED_SIGNATURES = {
    "representation": ["def build_forecast_inputs(", "def build_classification_inputs(", "class AllowedContextEncoder"],
    "objective_tier1": ["def forecast_loss(", "def compute_class_weights(", "def classification_loss("],
    "objective_tier2": ["def unified_pretrain_forward("],
}
FORBIDDEN_IMPORT_PAT = re.compile(
    r"^\s*(?:import|from)\s+(?:aerowf_|trial_features|trial_objective)", re.MULTILINE
)

# ---- objective_tier2：函数级替换协议 ----
# axis_lock（DEC-002）备注 unified_model.py 允改范围限 unified_pretrain_forward。
# 执行方式：LLM 只输出该方法的替换实现，驱动机械拼接回原文件——函数段之外
# 逐字节不可变由拼接构造保证（比整文件重写 + 事后 diff 审计更强的机器执法）。
TIER2_AXIS = "objective_tier2"
TIER2_FUNC = "unified_pretrain_forward"


def _tier2_func_span(source: str) -> tuple[int, int]:
    """定位 TIER2_FUNC 在 source 中的行跨度（1-based 闭区间）。"""
    import ast
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == TIER2_FUNC:
            start = node.lineno
            if node.decorator_list:
                start = min(d.lineno for d in node.decorator_list)
            return start, node.end_lineno
    raise ValueError(f"{TIER2_FUNC} not found in source")


def _normalize_func_lines(func_src: str) -> list[str] | None:
    """按 def 行的缩进精确平移到零缩进（不用 dedent，保留 docstring 内空行原样）。"""
    lines = func_src.splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    if not lines:
        return None
    base = len(lines[0]) - len(lines[0].lstrip())
    out: list[str] = []
    for ln in lines:
        if not ln.strip():
            out.append(ln[base:] if len(ln) > base else "")
        else:
            if len(ln) - len(ln.lstrip()) < base:
                return None
            out.append(ln[base:])
    return out


def _validate_tier2_function(func_src: str) -> str | None:
    """候选函数体校验：恰好一个同名函数、体内禁 import。返回错误或 None。"""
    import ast
    norm = _normalize_func_lines(func_src)
    if norm is None:
        return "tier2 edit has inconsistent indentation or is empty"
    try:
        tree = ast.parse("\n".join(norm))
    except SyntaxError as exc:
        return f"tier2 function does not parse: {exc}"
    fns = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    if len(tree.body) != 1 or len(fns) != 1 or fns[0].name != TIER2_FUNC:
        return f"tier2 edit must be exactly one function named {TIER2_FUNC}"
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            return "import statements are not allowed inside the tier2 function"
    return None


def splice_tier2(current: str, func_src: str) -> tuple[str | None, str | None]:
    """把候选方法拼回原文件（类内 4 空格缩进）。返回 (新全文, 错误)。"""
    err = _validate_tier2_function(func_src)
    if err:
        return None, err
    norm = _normalize_func_lines(func_src)
    indented = ["    " + ln if ln else "" for ln in norm]
    lines = current.splitlines()
    try:
        start, end = _tier2_func_span(current)
    except ValueError as exc:
        return None, str(exc)
    new_source = "\n".join(lines[: start - 1] + indented + lines[end:]) + "\n"
    try:
        _tier2_func_span(new_source)
    except (ValueError, SyntaxError) as exc:
        return None, f"spliced file invalid: {exc}"
    return new_source, None


def extract_tier2_function(source: str) -> str:
    """取出 TIER2_FUNC 的当前源码（保留类内缩进）。"""
    start, end = _tier2_func_span(source)
    return "\n".join(source.splitlines()[start - 1:end])


def _normalized_ast_dump(source: str) -> str:
    """AST 规范化摘要：剥离 docstring（注释本不入 AST）后 dump。
    用于 no-op 编辑检测——llm-obj-010 只改注释/docstring 就宣称完成 patch_plan，
    烧了整发 GPU 训出与 parent bit 级相同的模型。"""
    import ast
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                    and isinstance(body[0].value.value, str):
                node.body = body[1:] or [ast.Pass()]
    return ast.dump(tree)


def is_noop_edit(current_source: str, new_source: str) -> bool:
    """编辑与基线在剥离 docstring 后 AST 等价 → 语义 no-op，训练前拒绝。"""
    try:
        return _normalized_ast_dump(current_source) == _normalized_ast_dump(new_source)
    except SyntaxError:
        return False


# 假自由拦截（22 文档规则 2 的形式检查）：自由提案机制若与某模板实质等价，
# 拒绝并要求重提。启发式：模板 action_id 的机制词（去掉编号）在提案文本中
# 命中 >= max(2, 词数-1) 个即判等价。
_FAKE_FREE_STOPWORDS = {"loss", "based", "aware", "with"}


def fake_free_equivalent(trial: dict, actions: list[dict], axis: str) -> str | None:
    if not trial.get("is_free_proposal"):
        return None
    blob = " ".join(
        str(trial.get(k, "")) for k in ("action_id", "hypothesis", "patch_plan")
    ).lower()
    for act in actions:
        if act.get("axis") != axis:
            continue
        tokens = [t for t in act["action_id"].lower().split("-")[1:]
                  if t and t not in _FAKE_FREE_STOPWORDS]
        if not tokens:
            continue
        hits = sum(1 for t in tokens if t in blob)
        if hits >= max(2, len(tokens) - 1):
            return act["action_id"]
    return None

# guardrail 咨询性阈值（decision_policy v1.1 为 3-seed 确认口径；单 seed 筛选阶段
# 仅作 advisory 标注，不构成拒绝依据）
GUARDRAIL_ADVISORY_FLOOR = {
    "classification_macro_f1": -0.02,
    "classification_csi_macro": -0.05,
    "hazard_class_f1": -0.10,
}

# R/O-tier1 预训练 checkpoint 复用表（2026-08-28 批准）：这两轴的编辑不触及预训练
# 代码路径，重训与复用权重级等价（212/212 张量逐位一致，独立重训对账验证）。
# tier2 改预训练目标，必须全跑，不入此表。
REUSE_PRETRAIN_CKPTS = {
    42: "/root/autodl-tmp/aerowf_baseline/AeroWF/results/aerowf_unified_pretrain_full_formal_seed42_v1/checkpoints/best_model.pth",
    43: "/root/autodl-tmp/aerowf_downstream_v2/results/full_pipeline/seed43_v2/pretrain/checkpoints/best_model.pth",
    2027: "/root/autodl-tmp/aerowf_downstream_v2/results/full_pipeline/seed2027_v2/pretrain/checkpoints/best_model.pth",
    3407: "/root/autodl-tmp/aerowf_downstream_v2/results/full_pipeline/seed3407_v2/pretrain/checkpoints/best_model.pth",
    5519: "/root/autodl-tmp/aerowf_downstream_v2/results/full_pipeline/seed5519_v2/pretrain/checkpoints/best_model.pth",
}

# parent seed42 参考（C evaluator v1.0.2，2026-08-27 落盘于 discovery/parent_refs/）
PARENT_REFS = {
    "forecast_scratch": {"RMSE_macro_norm": 0.04847144659294747, "MAE_macro_norm": 0.0254475019647884},
    "forecast_pretrained": {"RMSE_macro_norm": 0.05090555, "MAE_macro_norm": 0.02848562},
    "classification_scratch": {"classification_macro_f1": 0.75035, "classification_csi_macro": 0.62679, "hazard_class_f1": 0.63309},
    "classification_pretrained": {"classification_macro_f1": 0.80365, "classification_csi_macro": 0.69388, "hazard_class_f1": 0.78857},
}

FAILURE_SLICES_SUMMARY = (
    "(a) 预训练对 forecast 是负迁移：3 个 seed（42/43/2027）上 scratch RMSE 一致优于 "
    "pretrained（seed42: 0.0485 vs 0.0509），配对 SNR 3.48，三 seed 同号；对 classification "
    "则是正迁移但幅度不稳定（macro_f1 增益 +0.077/+0.007 波动）。"
    "(b) 预测误差随时距单调恶化：seed43 scratch 分时距 RMSE T+1 0.0444 / T+4 0.0489 / T+8 0.0511，"
    "长时距（T+8, 120min）是最弱切片。"
    "(c) ZSSS 的 T+1 wind_x 出现回归头微溢出（>1 的预测值），提示低风速/边界值段拟合粗糙。"
    "(d) hazard 类 F1 跨 seed 波动极大（0.53~0.79，配对 SD 0.095），val 中 hazard 信号仅 ZBAA "
    "有支撑（support=94），ZSPD/ZSSS val 标签近单类。"
    "(e) 输入侧风用 (wind_x, wind_y) 归一化分量表示，低风速时方向信息被幅值淹没。"
)


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# LLM 端点与模型：由 CLI 配置（--model/--api-base），代码不写死具体型号。
# 每次调用把服务端实际返回的模型名收进 _SERVED_MODELS，随 trial 记录入 lineage
# （审计底账：论文须能如实交代 LLM 臂各阶段用了什么模型）。
LLM_MODEL = "deepseek-v4-pro"
LLM_API_BASE = "https://api.deepseek.com"
_SERVED_MODELS: set[str] = set()


def call_llm(api_key: str, messages: list[dict], temperature: float, max_tokens: int = 8000) -> str:
    body = json.dumps({
        "model": LLM_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{LLM_API_BASE}/chat/completions",
        data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    served = payload.get("model")
    if served:
        _SERVED_MODELS.add(str(served))
    return payload["choices"][0]["message"]["content"]


def load_lineage() -> list[dict]:
    if not LINEAGE.exists():
        return []
    return [json.loads(line) for line in LINEAGE.read_text(encoding="utf-8").splitlines() if line.strip()]


def append_lineage(record: dict) -> None:
    with LINEAGE.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


_SMOKE_SNIPPETS = {
    "representation": """
import importlib.util, sys
import numpy as np, torch
spec = importlib.util.spec_from_file_location("trial_mod", sys.argv[1])
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
# 槽位数用 4/5 两档，避开与时距(3)/分量(2)同形导致维度猜测型 bug 逃逸
for n_slots in (5, 4):
    runway = np.random.rand(96, n_slots, 2).astype(np.float32)
    mask = np.array([True] * (n_slots - 1) + [False])
    exo_cat = {"weather_code": 2, "sky_condition": 1, "has_gust": 0, "is_cavok": 1}
    exo_cont = np.array([0.5, 0.3, 0.0], dtype=np.float32)
    out = m.build_forecast_inputs(runway, mask, exo_cat, exo_cont, norm_stats=None)
    assert out["x"].shape == (96, n_slots, 2), f"forecast x shape {out['x'].shape}"
    assert out["node_mask"].dtype == torch.bool
    out2 = m.build_classification_inputs(runway, mask, exo_cat, exo_cont)
    assert out2["x"].shape == (96, n_slots, 2), f"cls x shape {out2['x'].shape}"
enc = m.AllowedContextEncoder(sky_known_max=5)
cat = {"sky_condition": torch.tensor([1, 2]), "has_gust": torch.tensor([0, 1]), "is_cavok": torch.tensor([1, 0])}
z = enc(cat, torch.rand(2, 3))
assert z.shape[0] == 2 and torch.isfinite(z).all()
print("SMOKE_OK")
""",
    "objective_tier1": """
import importlib.util, sys
import numpy as np, torch
spec = importlib.util.spec_from_file_location("trial_mod", sys.argv[1])
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
# 关键：槽位数不得等于时距数（3）或分量数（2），否则维度猜测型 bug 逃逸
# （llm-obj-003 教训：真实训练 slots=4，与 horizon=3 不同形）
for n_slots in (5, 4):
    pred = torch.rand(4, n_slots, 3, 2); target = torch.rand(4, n_slots, 3, 2)
    node_mask = torch.tensor([[1] * (n_slots - 1) + [0]] * 4, dtype=torch.bool)
    loss = m.forecast_loss(pred, target, node_mask)
    assert loss.dim() == 0 and torch.isfinite(loss), f"forecast_loss bad (slots={n_slots}): {loss}"
    pred.requires_grad_(True)
    loss2 = m.forecast_loss(pred, target, node_mask); loss2.backward()
    assert pred.grad is not None and torch.isfinite(pred.grad).all()
w = m.compute_class_weights(np.array([1000, 200, 50]))
assert w.shape == (3,) and torch.isfinite(w).all()
logits = torch.randn(8, 3, requires_grad=True)
label = torch.tensor([0, 1, 2, 0, 1, 2, 0, -100])
closs = m.classification_loss(logits, label, class_weights=w)
assert closs.dim() == 0 and torch.isfinite(closs); closs.backward()
print("SMOKE_OK")
""",
    "objective_tier2": """
import sys
sys.path.insert(0, "/root/autodl-tmp/aerowf_baseline/AeroWF")
import importlib.util
import torch
spec = importlib.util.spec_from_file_location("models.AirFM.unified_model_trial", sys.argv[1])
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)
torch.manual_seed(0)
# 缩小维度的真实构造：走与正式预训练相同的 unified_pretrain_forward 全路径
# （物理距离 GT + 混合掩码 + 编码 + 重建/对比双损失 + 反传）
config = {
    "Data_shape": (2, 4, 96, 11),
    "N_max": 4, "num_nodes": 3,
    "emb_size": 32, "rep_size": 64, "num_heads": 2, "dim_ff": 64,
    "dropout": 0.0,
    "encoder_num_heads": 2, "encoder_num_layers": 1, "encoder_dim_ff": 64,
    "frets_num_layers": 1,
    "use_simple_gnn": True, "gnn_hidden": 32, "gnn_layers": 1,
    "fusion_type": "residual_add", "use_frequency": True, "use_hierarchy": True,
    "use_masked_recon": True, "mask_ratio": 0.25,
    "random_mask_strategy": "random", "causal_mask_strategy": "last", "causal_prob": 0.5,
    "proj_hidden_dim": 32, "proj_output_dim": 16,
    "lambda_recon": 1.0, "lambda_contrast": 0.5,
    "ema_momentum": 0.99, "warmup_batches": 1,
    "sdtw_gamma": 0.1, "softdtw_pair_chunk_size": 8,
}
model = m.UnifiedSeries2Vec(config, num_classes=3)
model.train()
x = torch.rand(2, 4, 96, 11)
node_mask = torch.tensor([[True, True, True, False], [True, True, False, False]])
loss, loss_dict = model.unified_pretrain_forward(x, sdtw=None, node_mask=node_mask)
assert loss.dim() == 0 and torch.isfinite(loss), f"pretrain loss bad: {loss}"
loss.backward()
grads = [p.grad for p in model.parameters() if p.grad is not None]
assert grads, "no parameter received gradient"
assert all(torch.isfinite(g).all() for g in grads), "non-finite gradients"
assert isinstance(loss_dict, dict) and loss_dict, "loss_dict missing/empty"
print("SMOKE_OK")
""",
}


def functional_smoke(axis: str, source_path: Path) -> str | None:
    """CPU 假张量冒烟：签名可调用、形状/有限性/反传均正常。返回错误文本或 None。"""
    import subprocess as sp
    snippet = _SMOKE_SNIPPETS[axis]
    proc = sp.run(
        ["/root/miniconda3/bin/python", "-c", snippet, str(source_path)],
        capture_output=True, text=True, timeout=120,
    )
    if proc.returncode == 0 and "SMOKE_OK" in proc.stdout:
        return None
    return (proc.stderr or proc.stdout).strip()[-600:]


def stderr_tail(output_root_base: Path, run_id: str, n: int = 12) -> str:
    p = output_root_base / f"{run_id}.stderr"
    if not p.exists():
        return ""
    lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-n:])


def gen_tier2_edit(api_key: str, trial: dict, current_source: str) -> tuple[str | None, list[str]]:
    """tier2 第二段 LLM：只产出 unified_pretrain_forward 的替换实现，驱动拼接回原文件。"""
    func_src = extract_tier2_function(current_source)
    prompt = f"""You are the code-editing module of a closed-loop auto-research system.
Your approved research proposal for this trial:
{json.dumps({k: trial[k] for k in ('hypothesis', 'patch_plan', 'expected_effect', 'falsification')}, ensure_ascii=False, indent=1)}

You must implement the patch_plan by rewriting ONE METHOD of the pretraining model:
`unified_pretrain_forward` of class UnifiedSeries2Vec. Your output will be
mechanically spliced back into the original file — nothing outside this method
can change, and any attempt to change it is discarded.

INTERFACE CONTRACT (must hold, violating = trial fails before training):
- Method signature must stay exactly:
  def unified_pretrain_forward(self, x, exo_categorical=None, exo_continuous=None, sdtw=None, node_mask=None)
- Must return (loss, loss_dict): loss a finite scalar tensor on the graph, loss_dict a dict.
- NO import statements inside the method. Use names already available at module
  level: torch, F, generate_hybrid_mask, apply_mask, masked_mse_loss, and self.* members.
- The soft-DTW kernel is locked: you may change how it is invoked or weighted, not the kernel.
- mask_ratio (masked fraction) is frozen; the random:causal mix (causal_prob argument
  at the generate_hybrid_mask call) may be changed only if the patch_plan says so.
- Keep tensor shapes flowing through encode/decoder identical to the current implementation.

Current method source:
```python
{func_src}
```

Answer with the complete replacement method in ONE ```python code block and nothing
else. Top-level indentation of the method in your block may be zero; it will be
re-indented for the class body automatically."""
    errors: list[str] = []
    for attempt in range(2):
        try:
            raw = call_llm(api_key, [{"role": "user", "content": prompt}], temperature=0.2)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"tier2 codegen LLM call failed: {exc}")
            continue
        m = re.search(r"```python\n(.*?)```", raw, re.DOTALL)
        if not m:
            errors.append("tier2 codegen output missing ```python block")
            continue
        try:
            assert_no_hidden_tokens(m.group(1))
        except Exception as exc:  # noqa: BLE001
            errors.append(f"hidden token in tier2 codegen output: {exc}")
            continue
        new_source, splice_err = splice_tier2(current_source, m.group(1))
        if splice_err:
            errors.append(splice_err)
            continue
        if is_noop_edit(current_source, new_source):
            errors.append("no-op edit: change is comments/docstring only, patch_plan not implemented")
            continue
        if FORBIDDEN_IMPORT_PAT.search(new_source) and not FORBIDDEN_IMPORT_PAT.search(current_source):
            errors.append("forbidden import introduced by tier2 edit")
            continue
        tmp = Path("/tmp/_codegen_check.py")
        tmp.write_text(new_source, encoding="utf-8")
        try:
            py_compile.compile(str(tmp), doraise=True)
        except py_compile.PyCompileError as exc:
            errors.append(f"py_compile failed: {exc}")
            continue
        return new_source, errors
    return None, errors


def gen_code_edit(api_key: str, axis: str, trial: dict, current_source: str) -> tuple[str | None, list[str]]:
    """第二段 LLM：把 patch_plan 落成新文件全文。返回 (新全文 or None, 错误列表)。"""
    if axis == TIER2_AXIS:
        return gen_tier2_edit(api_key, trial, current_source)
    contract_text = INTERFACE_CONTRACTS.get(axis, "")
    prompt = f"""You are the code-editing module of a closed-loop auto-research system.
Your approved research proposal for this trial:
{json.dumps({k: trial[k] for k in ('hypothesis', 'patch_plan', 'expected_effect', 'falsification')}, ensure_ascii=False, indent=1)}

You must now implement the patch_plan by rewriting ONE file completely.

INTERFACE CONTRACT (must hold, violating = trial fails before training):
{contract_text}

Hard rules:
- Keep every public function/class signature listed in the contract unchanged.
- Do not import the locked training scripts or the other axis file.
- Keep tensor shapes and dtypes returned by each function identical to the current file.
- Pure-Python + numpy + torch only. No file I/O, no network, no subprocess.
- The change must implement exactly the approved patch_plan, nothing else.

Current file content:
```python
{current_source}
```

Answer with the complete new file content in ONE ```python code block and nothing else."""
    errors: list[str] = []
    for attempt in range(2):
        try:
            raw = call_llm(api_key, [{"role": "user", "content": prompt}], temperature=0.2)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"codegen LLM call failed: {exc}")
            continue
        m = re.search(r"```python\n(.*?)```", raw, re.DOTALL)
        if not m:
            errors.append("codegen output missing ```python block")
            continue
        source = m.group(1)
        try:
            assert_no_hidden_tokens(source)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"hidden token in codegen output: {exc}")
            continue
        missing = [sig for sig in REQUIRED_SIGNATURES[axis] if sig not in source]
        if missing:
            errors.append(f"missing required signatures: {missing}")
            continue
        if is_noop_edit(current_source, source):
            errors.append("no-op edit: change is comments/docstring only, patch_plan not implemented")
            continue
        if FORBIDDEN_IMPORT_PAT.search(source):
            errors.append("forbidden import of locked/other-axis module")
            continue
        tmp = Path("/tmp/_codegen_check.py")
        tmp.write_text(source, encoding="utf-8")
        try:
            py_compile.compile(str(tmp), doraise=True)
        except py_compile.PyCompileError as exc:
            errors.append(f"py_compile failed: {exc}")
            continue
        return source, errors
    return None, errors


def eval_all_legs(output_root: Path, trial_id: str, seed: int, out_dir: Path,
                  parent_refs: dict | None = None) -> tuple[dict, dict, str]:
    """四条腿评测。返回 (metrics_by_endpoint 扁平, policy_deltas, 综合状态)。"""
    flat: dict = {}
    deltas: dict = {}
    statuses: list[str] = []
    manifest_digest = ""
    for stage in ("forecast_scratch", "forecast_pretrained", "classification_scratch", "classification_pretrained"):
        task = "forecast" if stage.startswith("forecast") else "classification"
        npz = output_root / stage / "validation_predictions.npz"
        sub = out_dir / stage
        sub.mkdir(parents=True, exist_ok=True)
        try:
            adapt_stage_npz(npz, task, sub / "predictions.npz", COUNTS, allow_fill=False)
        except Exception as exc:  # noqa: BLE001
            flat[f"{stage}.__adapter_error__"] = str(exc)
            statuses.append("failed")
            continue
        meta = {"trial_id": trial_id, "arm": "llm", "seed": seed, "task": task, "checkpoint_digest": stage}
        (sub / "trial_meta.json").write_text(json.dumps(meta))
        ev = run_evaluator(EVAL_CMD, sub / "predictions.npz", sub / "eval_out", trial_meta_path=sub / "trial_meta.json")
        statuses.append(ev.status)
        if ev.status != "completed":
            flat[f"{stage}.__evaluator_status__"] = f"{ev.status}: {ev.status_reason[:200]}"
            continue
        pol = ev.raw_metrics.get("decision_policy_metrics", {})
        for key, entry in pol.items():
            flat[f"{stage}.policy.{key}"] = entry
            ref = (parent_refs or PARENT_REFS).get(stage, {}).get(key)
            if ref is not None:
                deltas[f"{stage}.{key}"] = round(entry["value"] - ref, 6)
        for name, ep in ev.metrics_by_endpoint.items():
            flat[f"{stage}.{name}"] = ep
        mpath = sub / "eval_out" / "evaluation_manifest.json"
        if mpath.exists():
            manifest_digest = hashlib.sha256(mpath.read_bytes()).hexdigest()
    if all(s == "completed" for s in statuses):
        combined = "completed"
    elif "failed" in statuses:
        combined = "failed"
    else:
        combined = "invalid"
    flat["__manifest_digest__"] = manifest_digest
    return flat, deltas, combined


def resolve_parent_refs(parent_trial: str) -> dict:
    """父子链：parent 为已完成 trial 时，用其评测指标作配对参考。"""
    if parent_trial == "parent-seed42-formal":
        return PARENT_REFS
    result_path = DISCOVERY / "trials" / parent_trial / "result.json"
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    if payload.get("status") != "completed":
        raise ValueError(f"parent trial {parent_trial} 状态非 completed，不能作为配对基准")
    refs: dict = {}
    for key, entry in payload.get("metrics_by_endpoint", {}).items():
        # 形如 "<stage>.policy.<metric>"
        if ".policy." in key and isinstance(entry, dict) and "value" in entry:
            stage, metric = key.split(".policy.", 1)
            refs.setdefault(stage, {})[metric] = entry["value"]
    if not refs:
        raise ValueError(f"parent trial {parent_trial} 无 policy 指标可作参考")
    return refs


TIER2_PRIMARY_KEY = "tier2_primary_pretrained_vs_parent_scratch.RMSE_macro_norm"


def adjudicate(result: dict, axis: str = "") -> tuple[str, str]:
    """裁决规则（预注册，筛选期单 seed 口径）。

    decision_policy v1.2（2026-08-28 C+A 双签，SHA 7268682a…）stage 绑定：
    - 默认轴：primary = forecast_scratch.RMSE_macro_norm 同 stage 配对（与 v1.1 实现一致）；
    - objective_tier2：primary = candidate forecast_pretrained vs parent forecast_scratch
      （负迁移修复到反超裸训）；同 stage 改善但未反超记 partial_effect（不 过筛）。
    阈值沿用 0.0005 不变。verdict 枚举受冻结 schema 限制，v1.2 细分标签写入 basis 文本。
    """
    if result["status"] != "completed":
        return "not_evaluated", "verdict_rule: 非 completed 不裁决"
    deltas = result["paired_delta_vs_parent"]
    if axis == TIER2_AXIS:
        primary = deltas.get(TIER2_PRIMARY_KEY)
        if primary is None:
            return "inconclusive", "verdict_rule_v1.2(tier2): 缺 pretrained-vs-parent-scratch 配对值"
        if primary <= -0.0005:
            return "supported", (
                f"verdict_rule_v1.2(tier2): pretrained 反超 parent scratch {-primary:.4f} >= 0.0005"
                "（单 seed，待确认）"
            )
        same_stage = deltas.get("forecast_pretrained.RMSE_macro_norm")
        if same_stage is not None and same_stage <= -0.0005:
            return "refuted", (
                f"verdict_rule_v1.2(tier2): partial_effect——负迁移收窄（同 stage Δ {same_stage:+.4f}）"
                f"但未反超 scratch（vs parent scratch Δ {primary:+.4f}），不构成 screen_pass"
            )
        return "refuted", (
            f"verdict_rule_v1.2(tier2): refuted_on_pretrained——vs parent scratch Δ {primary:+.4f}，"
            f"同 stage Δ {same_stage:+.4f}（无改善或恶化）"
        )
    primary = deltas.get("forecast_scratch.RMSE_macro_norm")
    if primary is None:
        return "inconclusive", "verdict_rule_v1: 缺主指标配对值"
    if primary <= -0.0005:
        return "supported", f"verdict_rule_v1: 主指标配对改善 {-primary:.4f} >= 筛选线 0.0005（单 seed，待确认）"
    return "refuted", f"verdict_rule_v1: 主指标配对 Δ {primary:+.4f} 未达筛选线（单 seed 无可测效应）"


def guardrail_advisory(deltas: dict) -> dict:
    """单 seed 咨询性 guardrail 标注（不构成拒绝）。"""
    breaches = []
    for key, delta in deltas.items():
        for metric, floor in GUARDRAIL_ADVISORY_FLOOR.items():
            if key.endswith(metric) and delta < floor:
                breaches.append(f"{key}: {delta:+.4f} < {floor}")
    return {
        "mode": "advisory_single_seed",
        "breaches": breaches or ["none"],
        "note": "正式 guardrail 判定在 3-seed 确认阶段执行（decision_policy v1.1）",
    }


def run_one_trial(api_key: str, axis: str, trial_seq: int, seed: int,
                  parent_trial: str = "parent-seed42-formal") -> dict:
    parent_refs = resolve_parent_refs(parent_trial)
    lineage = load_lineage()
    try:
        failure_slices = build_failure_slices(DISCOVERY / "parent_refs")
    except Exception:  # noqa: BLE001
        failure_slices = FAILURE_SLICES_SUMMARY
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    active_ids = [a["action_id"] for a in registry["actions"]
                  if a.get("status") == "active" and a["axis"] == axis]
    forced, forced_reason = forced_free_status(lineage, axis, active_ids)
    prompt = assemble_proposal_prompt(
        axis=axis,
        registry_path=REGISTRY,
        lineage_records=lineage,
        failure_slices_summary=failure_slices,
    )
    if forced:
        prompt += FORCED_FREE_DIRECTIVE
        print(f"[forced-free] {axis}: {forced_reason}", flush=True)
    assert_no_hidden_tokens(prompt)

    trial = None
    proposal_errors: list[str] = []
    for attempt in range(3):
        raw = call_llm(api_key, [{"role": "user", "content": prompt}], temperature=0.7)
        parsed = parse_llm_proposal(raw, trial_seq=trial_seq, parent_trial=parent_trial, screening_seed=seed)
        if parsed.ok and forced and not parsed.trial_record["is_free_proposal"]:
            proposal_errors.append(
                f"强制自由轮拒绝模板提案: {parsed.trial_record['action_id']}（{forced_reason}）"
            )
            continue
        if parsed.ok and parsed.trial_record["axis"] == axis:
            equiv = fake_free_equivalent(parsed.trial_record, registry["actions"], axis)
            if equiv:
                proposal_errors.append(
                    f"假自由拦截（22 文档规则 2）: {parsed.trial_record['action_id']} "
                    f"与模板 {equiv} 实质等价，应改用该模板或提出模板外机制"
                )
                continue
            trial = parsed.trial_record
            trial["forced_free_round"] = forced
            break
        proposal_errors.extend(parsed.errors or [f"axis mismatch: {parsed.trial_record.get('axis')}"])
    if trial is None:
        rec = {"event": "proposal_rejected", "axis": axis, "trial_seq": trial_seq,
               "forced_free_round": forced,
               "errors": proposal_errors[:6], "at": now_utc()}
        append_lineage(rec)
        return rec

    trial_dir = DISCOVERY / "trials" / trial["trial_id"]
    trial_dir.mkdir(parents=True, exist_ok=True)
    (trial_dir / "trial.json").write_text(json.dumps(trial, ensure_ascii=False, indent=2), encoding="utf-8")

    target = AXIS_FILES[axis]
    current = target.read_text(encoding="utf-8")
    new_source, gen_errors = gen_code_edit(api_key, axis, trial, current)
    if new_source is None:
        rec = {"event": "codegen_rejected", "trial_id": trial["trial_id"], "axis": axis,
               "hypothesis": trial["hypothesis"], "errors": gen_errors[:6], "at": now_utc()}
        append_lineage(rec)
        return rec
    (trial_dir / "edit.py").write_text(new_source, encoding="utf-8")

    # CPU 冒烟闸门（G-15）：假张量调用全部接口，坏编辑不烧 GPU；一次修复机会
    smoke_err = functional_smoke(axis, trial_dir / "edit.py")
    if smoke_err:
        if axis == TIER2_AXIS:
            repair_payload = extract_tier2_function(new_source)
            repair_unit = "method (splice protocol unchanged)"
        else:
            repair_payload = new_source
            repair_unit = "file"
        repair_prompt = (
            "Your previous code edit failed a functional smoke test before training.\n"
            f"Error:\n{smoke_err}\n\n"
            "Fix the bug. Keep the same approved patch_plan and all interface signatures. "
            f"Answer with the complete corrected {repair_unit} in ONE ```python block and nothing else.\n\n"
            f"Your previous content:\n```python\n{repair_payload}\n```"
        )
        try:
            raw = call_llm(api_key, [{"role": "user", "content": repair_prompt}], temperature=0.2)
            m = re.search(r"```python\n(.*?)```", raw, re.DOTALL)
            if m:
                block = m.group(1)
                assert_no_hidden_tokens(block)
                candidate = None
                if axis == TIER2_AXIS:
                    candidate, splice_err = splice_tier2(current, block)
                    if splice_err:
                        candidate = None
                elif not FORBIDDEN_IMPORT_PAT.search(block) and all(
                    sig in block for sig in REQUIRED_SIGNATURES[axis]
                ):
                    candidate = block
                if candidate is not None:
                    (trial_dir / "edit.py").write_text(candidate, encoding="utf-8")
                    smoke_err = functional_smoke(axis, trial_dir / "edit.py")
                    if smoke_err is None:
                        new_source = candidate
        except Exception as exc:  # noqa: BLE001
            smoke_err = f"repair round failed: {exc}"
    if smoke_err:
        rec = {"event": "smoke_rejected", "trial_id": trial["trial_id"], "axis": axis,
               "hypothesis": trial["hypothesis"], "smoke_error": smoke_err[:400], "at": now_utc()}
        append_lineage(rec)
        return rec

    if axis == TIER2_AXIS:
        # axis_lock 备注的人审兜底材料：函数级 diff 落盘（advisory，不阻塞）
        import difflib
        diff = "\n".join(difflib.unified_diff(
            extract_tier2_function(current).splitlines(),
            extract_tier2_function(new_source).splitlines(),
            fromfile=f"baseline/{TIER2_FUNC}", tofile=f"{trial['trial_id']}/{TIER2_FUNC}",
            lineterm="",
        ))
        (trial_dir / "tier2_function_diff.txt").write_text(diff + "\n", encoding="utf-8")

    lock_cfg = load_config(CONTRACT / "axis_lock_v1_draft.json", AEROWF_REPO, DOWNSTREAM)
    dec = check([str(target)], axis, lock_cfg)
    if not dec.allowed:
        rec = {"event": "axis_lock_rejected", "trial_id": trial["trial_id"], "axis": axis,
               "violations": dec.violations, "at": now_utc()}
        append_lineage(rec)
        return rec

    backup = target.with_suffix(".py.discovery_backup")
    shutil.copy2(target, backup)
    started = time.monotonic()
    try:
        target.write_text(new_source, encoding="utf-8")
        cfg = PipelineConfig(
            pipeline_script=Path(DOWNSTREAM) / "src/aerowf_full_pipeline_v2.py",
            workdir=Path(AEROWF_REPO),
            output_root_base=Path(DOWNSTREAM) / "results/harness",
            python_bin="/root/miniconda3/bin/python",
            pretrain_epochs=100,
            downstream_epochs=30,
            timeout_sec=6 * 3600,
        )
        extra_args: list[str] = []
        if axis != TIER2_AXIS and seed in REUSE_PRETRAIN_CKPTS:
            extra_args = ["--reuse-pretrain-checkpoint", REUSE_PRETRAIN_CKPTS[seed]]
        outcome = run_pipeline(cfg, seed=seed,
                               run_id=f"disc_{trial['trial_id'].replace('-', '_')}_seed{seed}",
                               extra_args=extra_args)

        result = {
            "trial_id": trial["trial_id"],
            "attempt_id": 1,
            "status": "failed",
            "status_reason": "; ".join(outcome.failure_reasons)[:500],
            "metrics_by_endpoint": {},
            "paired_delta_vs_parent": {},
            "artifact_digests": {"edit_sha256": sha256_text(new_source),
                                 **{f"ckpt.{k}": v for k, v in outcome.checkpoint_sha256.items()}},
            "resource_usage": {"gpu_hours": 0.0, "gpu_model": "RTX3090"},
            "hypothesis_verdict": "not_evaluated",
            "execution_deviations": [],
            "evaluation_manifest_digest": "",
            "created_at_utc": now_utc(),
        }
        if outcome.status == "success":
            flat, deltas, combined = eval_all_legs(
                outcome.output_root, trial["trial_id"], seed, trial_dir / "eval",
                parent_refs=parent_refs,
            )
            if axis == TIER2_AXIS:
                # decision_policy v1.2 stage 绑定：tier2 primary = pretrained vs parent scratch
                pre = flat.get("forecast_pretrained.policy.RMSE_macro_norm")
                scr_ref = parent_refs.get("forecast_scratch", {}).get("RMSE_macro_norm")
                if isinstance(pre, dict) and "value" in pre and scr_ref is not None:
                    deltas[TIER2_PRIMARY_KEY] = round(pre["value"] - scr_ref, 6)
            result["status"] = combined
            result["status_reason"] = "screening 单 seed 完成" if combined == "completed" else "评测未全部 completed"
            result["metrics_by_endpoint"] = {k: v for k, v in flat.items() if not k.startswith("__")}
            result["paired_delta_vs_parent"] = deltas
            result["evaluation_manifest_digest"] = flat.get("__manifest_digest__", "")
            result["guardrail_check"] = guardrail_advisory(deltas)
            primary_key = TIER2_PRIMARY_KEY if axis == TIER2_AXIS else "forecast_scratch.RMSE_macro_norm"
            primary = deltas.get(primary_key)
            if primary is not None:
                # decision_policy v1.2 筛选线：配对改善 >= 0.0005（stage 绑定按轴）
                result["screen_pass"] = bool(primary <= -0.0005)
        verdict, verdict_basis = adjudicate(result, axis=axis)
        result["hypothesis_verdict"] = verdict
        tail = ""
        if outcome.status != "success":
            tail = stderr_tail(cfg.output_root_base, outcome.run_id)
        result["resource_usage"]["gpu_hours"] = round((time.monotonic() - started) / 3600, 3)
        validate_result({k: v for k, v in result.items() if k != "screen_pass"})
        (trial_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        append_lineage({
            "event": "trial_done", "trial_id": trial["trial_id"], "axis": axis,
            "llm_model_config": LLM_MODEL,
            "llm_model_served": sorted(_SERVED_MODELS),
            "action_id": trial["action_id"], "parent_trial": parent_trial,
            "arm_category": "llm_free" if trial.get("is_free_proposal") else "llm_template",
            "non_expressibility": trial.get("non_expressibility", ""),
            "forced_free_round": trial.get("forced_free_round", False),
            "hypothesis": trial["hypothesis"],
            "falsification": trial["falsification"], "status": result["status"],
            "status_reason": result["status_reason"],
            "stderr_tail": tail[:400],
            "paired_delta_vs_parent": result["paired_delta_vs_parent"],
            "screen_pass": result.get("screen_pass"),
            "hypothesis_verdict": verdict, "verdict_basis": verdict_basis,
            "guardrail_advisory": result.get("guardrail_check", {}).get("breaches"),
            "gpu_hours": result["resource_usage"]["gpu_hours"], "at": now_utc(),
        })
        return result
    finally:
        shutil.copy2(backup, target)
        backup.unlink()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api-key", required=True)
    ap.add_argument("--plan", required=True, help="逗号分隔的轴序列，如 representation,objective_tier1")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--start-seq", type=int, default=0)
    ap.add_argument("--parent-trial", default="parent-seed42-formal",
                    help="配对基准：默认冻结 parent；传已完成 trial_id 则形成父子链")
    ap.add_argument("--model", default=None,
                    help="LLM 模型名（默认 deepseek-v4-pro；切换须先于任何 screen_pass 并入 lineage 留痕）")
    ap.add_argument("--api-base", default=None, help="OpenAI 兼容端点根地址")
    args = ap.parse_args()
    if args.model:
        globals()["LLM_MODEL"] = args.model
    if args.api_base:
        globals()["LLM_API_BASE"] = args.api_base.rstrip("/")

    DISCOVERY.mkdir(parents=True, exist_ok=True)
    # 冒烟闸门自检：基线文件必须通过，否则是闸门自身的假张量形状错了
    for axis, path in AXIS_FILES.items():
        err = functional_smoke(axis, path)
        if err:
            print(f"SMOKE_GATE_SELFCHECK_FAILED axis={axis}: {err}", flush=True)
            return 1
    print("smoke gate 自检通过（基线文件）", flush=True)
    axes = [a.strip() for a in args.plan.split(",") if a.strip()]
    for i, axis in enumerate(axes):
        seq = args.start_seq + i
        print(f"\n===== TRIAL {seq} axis={axis} {now_utc()} =====", flush=True)
        try:
            out = run_one_trial(args.api_key, axis, seq, args.seed,
                                parent_trial=args.parent_trial)
        except Exception as exc:  # noqa: BLE001
            print(f"TRIAL_DRIVER_ERROR: {exc}", flush=True)
            append_lineage({"event": "driver_error", "axis": axis, "trial_seq": seq,
                            "error": str(exc)[:300], "at": now_utc()})
            continue
        print("TRIAL_OUTCOME:", json.dumps(
            {k: out.get(k) for k in ("trial_id", "event", "status", "paired_delta_vs_parent",
                                     "screen_pass", "hypothesis_verdict")},
            ensure_ascii=False), flush=True)
    print("\nBATCH_DONE", now_utc(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
