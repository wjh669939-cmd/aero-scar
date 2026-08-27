"""A-3: 随机对照臂采样器。

从 action registry 中 status=active 且带 param_space 的模板动作里
均匀合法采样，生成符合 trial.schema.json 的 trial 记录。

设计要点（05 文档 §七）:
- 只采模板动作（parameterizable + 预实现的 code_level 模板），不含 LLM 自由提案；
- 与 LLM 臂共用 schema / evaluator / DecisionPolicy / 预算；
- 采样过程可复现：给定 seed，序列确定。

用法:
    python -m contract_tools.random_arm --n 15 --axis representation --seed 20260901 --out-dir out/
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path

from contract_tools.validate import validate_trial

CONTRACT_DIR = Path(__file__).resolve().parents[2] / "00_contract"
REGISTRY_PATH = CONTRACT_DIR / "action_registry_v1_draft.json"

_AXIS_SHORT = {
    "representation": "rep",
    "objective_tier1": "obj",
    "objective_tier2": "obj",
    "model": "model",
}


def load_registry(path: Path = REGISTRY_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sampleable_actions(registry: dict, axis: str | None = None) -> list[dict]:
    out = []
    for action in registry["actions"]:
        if action.get("status") != "active":
            continue
        if axis is not None and action["axis"] != axis:
            continue
        out.append(action)
    return out


def _sample_params(param_space: dict, rng: random.Random) -> dict:
    params = {}
    for name, spec in param_space.items():
        kind = spec["kind"]
        choices = spec["choices"]
        if kind == "choice":
            params[name] = rng.choice(choices)
        elif kind == "subset_nonempty":
            k = rng.randint(1, len(choices))
            params[name] = sorted(rng.sample(choices, k), key=str)
        else:
            raise ValueError(f"unknown param_space kind: {kind}")
    return params


def sample_trials(
    *,
    n: int,
    seed: int,
    axis: str | None = None,
    parent_trial: str = "parent-scratch-5seed",
    gpu_hours_cap: float = 1.0,
    registry: dict | None = None,
) -> list[dict]:
    registry = registry or load_registry()
    pool = sampleable_actions(registry, axis)
    if not pool:
        raise ValueError(f"no active sampleable actions for axis={axis!r}")
    rng = random.Random(seed)
    screening_seeds = json.loads((CONTRACT_DIR / "seeds.json").read_text(encoding="utf-8"))["screening_seeds"]
    trials = []
    for i in range(n):
        action = rng.choice(pool)
        params = _sample_params(action.get("param_space", {}), rng)
        short = _AXIS_SHORT[action["axis"]]
        trial = {
            "trial_id": f"rand-{short}-{i:03d}",
            "arm": "random",
            "axis": action["axis"],
            "tier": action["tier"],
            "parent_trial": parent_trial,
            "action_id": action["action_id"],
            "is_free_proposal": False,
            "hypothesis": action["hypothesis"],
            "evidence_anchor": action["evidence_anchor"],
            "target_slices": action["target_slices"],
            "expected_effect": f"随机臂镜像动作，参数 {json.dumps(params, ensure_ascii=False)}",
            "falsification": action["falsification"],
            "editable_paths": _editable_paths(registry, action["axis"]),
            "sampled_params": params,
            "budget": {"gpu_hours_cap": gpu_hours_cap, "seeds": screening_seeds},
            "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        validate_trial(trial)
        trials.append(trial)
    return trials


def _editable_paths(registry: dict, axis: str) -> list[str]:
    key = axis if axis in registry["axes"] else axis.split("_")[0]
    spec = registry["axes"].get(key) or registry["axes"][axis]
    return spec["editable_paths_placeholder"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--axis", default=None)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()
    trials = sample_trials(n=args.n, seed=args.seed, axis=args.axis)
    if args.out_dir:
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        for t in trials:
            (out / f"{t['trial_id']}.json").write_text(
                json.dumps(t, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        print(f"wrote {len(trials)} trials to {out}")
    else:
        print(json.dumps(trials, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
