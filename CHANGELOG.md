# Changelog

All notable states of this archive are documented here.
Component versions are frozen contracts: any change requires a version bump,
golden/report regeneration where applicable, and a written rationale in `docs/`.

## unreleased — 2026-08-28 (pre-push archive)

Docs and governance catch-up; no harness/evaluator behavior change.

- Add plan docs 17/18/19 (C private deploy req, 8/27 standup, G-10 joint-debug) and 25 (Axis Profile scheme for meeting).
- Update docs/iclr2027_plan/14_完成清单/README.md org note: C private dir clones with image (no per-machine redeploy).
- Add contract/decision_policy_v1.1_SIGNOFF.md (C+A frozen 2026-08-27).
- Add docs/model_side/g8_seed43_consistency_report.md.
- Add discovery/parent_refs_metrics/seed3407_summary.md (metrics only; seed5519 pending).

## v0.1.0-pre-discovery — 2026-08-27

First archived snapshot. All pre-discovery gates (G-1 .. G-14) closed;
formal discovery batch02 (2 LLM-arm screening trials, seed 42) running at snapshot time.

### Component versions at this snapshot

| Component | Version | Evidence / Digest |
|---|---|---|
| C1 evaluator | **v1.0.2** (frozen 2026-08-27) | config `evaluator_config_v1.json` SHA-256 `e52c8404…`; full file list in `evaluator/C1_evaluator_current/reports/SHA256摘要.json` |
| Downstream task contract | **v2.0** | `contract/DOWNSTREAM_TASK_CONTRACT_v2.json` |
| decision_policy | **v1.1** (thresholds calibrated from 3-seed variance report, frozen before any candidate existed) | `contract/decision_policy_v1_draft.json` |
| axis_lock | **v1.0 frozen** + DEC-002 amendment (objective_tier2 activated 2026-08-27) | `contract/axis_lock_v1_draft.json` |
| action registry | v1 draft, O4-O6 active per DEC-002; M1-M3 conditional per DEC-001 | `contract/action_registry_v1_draft.json` |
| seeds | **v1.1** (ratified; screening 42, confirmation {42,43,2027}, final +{3407,5519}) | `contract/seeds.json` |
| test lock | LOCKED (temporal / spatial / event); one-shot unlock window 2026-09-11..12 | `contract/test_lock_state.json` (redacted contract copy) |
| CLH harness | P0 subprocess isolation fix + R/O axis config; all unit tests green | `harness/` |
| contract tools | 55 unit tests green on both machines | `tools/` |
| discovery runner | two-stage LLM (proposal JSON -> code edit) + CPU functional smoke gate + repair round | `discovery/discovery_runner.py` |

### Baseline references (frozen evaluator v1.0.2, validation only)

- parent seed42 forecast scratch: RMSE_macro_norm **0.048471**, MAE_macro_norm 0.025448
- parent seed42 forecast pretrained: RMSE_macro_norm 0.050906 (negative transfer, 3-seed consistent)
- parent seed42 classification scratch / pretrained macro-F1: 0.75035 / 0.80365

### Known events recorded in lineage

- `llm-rep-000`, `llm-obj-001`: failed on LLM code bugs (UnboundLocalError; broadcast
  shape mismatch) after pretrain stage; root cause of the missing pre-training smoke
  gate fixed in the runner afterwards (gate now intercepts both historical edits in
  seconds on CPU). Kept in lineage as honest negative records.

### Intentionally excluded from this repository

- frozen data deliveries (`AeroWF_v1_MODEL_TRAINING` / `EVALUATION`), all npy/npz
  training data, model weights (`*.pth`), training output trees (`results/`);
- C-side private evaluator config (validation ground truth, private masks),
  certification data, lock/access logs (only SHA-256 reconciliation lists kept);
- API keys and machine credentials (injected at runtime, never written to disk).
