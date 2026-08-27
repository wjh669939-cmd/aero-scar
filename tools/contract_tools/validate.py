"""A-1: trial/result 记录的 schema 校验入口。

用法（库）:
    from contract_tools.validate import validate_trial, validate_result
用法（CLI）:
    python -m contract_tools.validate trial path/to/trial.json
"""

from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path

import jsonschema

CONTRACT_DIR = Path(__file__).resolve().parents[2] / "00_contract"
SCHEMA_DIR = CONTRACT_DIR / "schemas"


class ContractViolation(Exception):
    """记录不符合冻结 schema。消息包含全部违规项，供闭环写入 lineage。"""


@lru_cache(maxsize=None)
def _schema(name: str) -> dict:
    return json.loads((SCHEMA_DIR / f"{name}.schema.json").read_text(encoding="utf-8"))


def _validate(record: dict, schema_name: str) -> None:
    validator = jsonschema.Draft202012Validator(_schema(schema_name))
    errors = sorted(validator.iter_errors(record), key=lambda e: list(e.absolute_path))
    if errors:
        lines = [
            f"{'/'.join(str(p) for p in e.absolute_path) or '<root>'}: {e.message}"
            for e in errors
        ]
        raise ContractViolation(
            f"{schema_name} record failed {len(errors)} check(s):\n" + "\n".join(lines)
        )


def validate_trial(record: dict) -> None:
    _validate(record, "trial")


def validate_result(record: dict) -> None:
    _validate(record, "result")


def check_param_budget(
    baseline_param_count: int,
    candidate_param_count: int,
    max_delta_ratio: float = 0.05,
) -> None:
    """M 轴反容量混淆 gate：参数量变化超预算即拒绝（05 文档 §八·五 约束 1）。"""
    if baseline_param_count <= 0:
        raise ContractViolation("baseline_param_count must be positive")
    ratio = abs(candidate_param_count - baseline_param_count) / baseline_param_count
    if ratio > max_delta_ratio:
        raise ContractViolation(
            f"param delta ratio {ratio:.4f} exceeds budget {max_delta_ratio:.4f} "
            f"(baseline={baseline_param_count}, candidate={candidate_param_count})"
        )


def main(argv: list[str]) -> int:
    if len(argv) != 2 or argv[0] not in {"trial", "result"}:
        print("usage: python -m contract_tools.validate {trial|result} <record.json>")
        return 2
    record = json.loads(Path(argv[1]).read_text(encoding="utf-8"))
    try:
        _validate(record, argv[0])
    except ContractViolation as exc:
        print(str(exc))
        return 1
    print(f"OK: {argv[1]} conforms to {argv[0]} schema")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
