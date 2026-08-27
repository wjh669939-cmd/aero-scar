"""A-11：lineage 统计与多机合并工具（纯读写，不触碰任何方法学规则）。

- stats：从 lineage.jsonl 生成 campaign 统计（站会日报 / 论文数字的唯一出处）；
- merge：多机并行（三轴三机）后的 lineage 合并——按记录哈希去重、按时间排序、
  trial_id 冲突即报错（防止序号分段约定被违反）。
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path


def load_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _record_key(rec: dict) -> str:
    return hashlib.sha256(json.dumps(rec, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def merge_lineages(paths: list[Path]) -> list[dict]:
    """合并多机 lineage：全量去重 + 按 at 排序 + trial_done 的 trial_id 唯一性检查。

    同一 trial_id 出现两条不同的 trial_done 记录 => 序号分段约定被违反，直接抛错。
    """
    seen: set[str] = set()
    merged: list[dict] = []
    trial_done_ids: dict[str, str] = {}
    for path in paths:
        for rec in load_records(path):
            key = _record_key(rec)
            if key in seen:
                continue
            seen.add(key)
            if rec.get("event") == "trial_done":
                tid = rec.get("trial_id", "")
                if tid in trial_done_ids and trial_done_ids[tid] != key:
                    raise ValueError(
                        f"trial_id 冲突: {tid} 在多台机器出现不同的 trial_done 记录"
                        "（检查 --start-seq 分段约定）"
                    )
                trial_done_ids[tid] = key
            merged.append(rec)
    merged.sort(key=lambda r: r.get("at", ""))
    return merged


def write_lineage(records: list[dict], path: Path) -> None:
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records),
        encoding="utf-8",
    )


def campaign_stats(records: list[dict]) -> dict:
    """campaign 统计：论文与站会数字的机器出处。"""
    events = Counter(r.get("event", "?") for r in records)
    done = [r for r in records if r.get("event") == "trial_done"]
    verdict_backfill = {
        r["trial_id"]: r.get("hypothesis_verdict")
        for r in records if r.get("event") == "verdict_backfill" and r.get("trial_id")
    }
    by_axis: dict[str, Counter] = {}
    by_category = Counter()
    verdicts = Counter()
    screen_passes: list[str] = []
    gpu_hours = 0.0
    for r in done:
        axis = r.get("axis", "?")
        by_axis.setdefault(axis, Counter())[r.get("status", "?")] += 1
        by_category[r.get("arm_category", "llm_template")] += 1
        v = verdict_backfill.get(r.get("trial_id"), r.get("hypothesis_verdict") or "not_evaluated")
        verdicts[v] += 1
        if r.get("screen_pass"):
            screen_passes.append(r.get("trial_id", "?"))
        gpu_hours += float(r.get("gpu_hours") or 0.0)
    return {
        "total_records": len(records),
        "events": dict(events),
        "trials_done": len(done),
        "by_axis_status": {a: dict(c) for a, c in sorted(by_axis.items())},
        "by_arm_category": dict(by_category),
        "verdicts": dict(verdicts),
        "screen_passes": screen_passes,
        "gpu_hours_total": round(gpu_hours, 2),
    }


def format_stats(stats: dict) -> str:
    lines = [
        f"trial 完成 {stats['trials_done']} 个 | GPU {stats['gpu_hours_total']}h | "
        f"过筛 {len(stats['screen_passes'])} 个 {stats['screen_passes'] or ''}",
        f"裁决: {stats['verdicts']}",
        f"臂类别: {stats['by_arm_category']}",
        f"分轴状态: {stats['by_axis_status']}",
        f"全部事件: {stats['events']}",
    ]
    return "\n".join(lines)


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_stats = sub.add_parser("stats")
    p_stats.add_argument("lineage", type=Path)
    p_merge = sub.add_parser("merge")
    p_merge.add_argument("inputs", type=Path, nargs="+")
    p_merge.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    if args.cmd == "stats":
        print(format_stats(campaign_stats(load_records(args.lineage))))
    else:
        merged = merge_lineages(args.inputs)
        write_lineage(merged, args.out)
        print(f"merged {len(args.inputs)} files -> {args.out} ({len(merged)} records)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
