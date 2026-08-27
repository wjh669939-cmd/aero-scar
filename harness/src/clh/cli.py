"""CLI for the closed-loop auto research harness."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console
from rich.table import Table

from clh.config import load_config
from clh.plugins.compose import boot_context
from clh.research.loop import ClosedLoopResearcher

console = Console()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="clh", description="Closed-loop Auto Research harness")
    sub = parser.add_subparsers(dest="cmd", required=True)
    run_p = sub.add_parser("run", help="run axis-isolated closed-loop search then certify")
    run_p.add_argument("--config", type=Path, default=Path("configs/dummy.toml"))
    run_p.add_argument("--run-id", default="")
    run_p.add_argument("--provider", choices=["offline", "openai_compat"], default="")
    status_p = sub.add_parser("status", help="print a run summary")
    status_p.add_argument("--run", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.cmd == "run":
        return _run(args)
    return _status(args.run)


def _run(args: argparse.Namespace) -> int:
    workspace = Path.cwd()
    config = load_config(args.config)
    if args.provider:
        config.llm.provider = args.provider
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = (workspace / config.run_root / run_id).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.resolved.json").write_text(
        config.model_dump_json(indent=2), encoding="utf-8"
    )
    ctx = boot_context(config, run_dir, workspace_root=workspace)
    console.print(f"[bold]CLH[/bold] profile={config.profile} llm={ctx.get('llm').name} run={run_dir}")
    researcher = ClosedLoopResearcher(ctx)
    summary = researcher.run()
    table = Table(title="Closed-loop trials")
    table.add_column("trial")
    table.add_column("axis")
    table.add_column("status")
    table.add_column("val Δ")
    for trial in researcher.trials:
        table.add_row(trial.trial_id, trial.axis, trial.status, f"{trial.improvement:+.3f}")
    console.print(table)
    console.print(f"certification → {run_dir / 'certification.json'}")
    console.print(f"summary → {run_dir / 'summary.json'}")
    del summary
    return 0


def _status(run_dir: Path) -> int:
    summary = run_dir / "summary.json"
    if not summary.exists():
        console.print(f"no summary.json in {run_dir}")
        return 1
    console.print(summary.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
