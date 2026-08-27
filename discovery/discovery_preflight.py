"""开跑前预检：只走 提案→解析→代码生成→axis_lock，不动 GPU、不装编辑。"""

import json
import sys

sys.path.insert(0, "/root/autodl-tmp/clh_deploy/discovery")
sys.path.insert(0, "/root/autodl-tmp/clh_deploy/workspace/runs/aerowf-v1/tools")

import discovery_runner as dr

api_key = sys.argv[1]
for axis in ("representation", "objective_tier1"):
    print(f"\n===== PREFLIGHT {axis} =====")
    lineage = dr.load_lineage()
    prompt = dr.assemble_proposal_prompt(
        axis=axis, registry_path=dr.REGISTRY,
        lineage_records=lineage, failure_slices_summary=dr.FAILURE_SLICES_SUMMARY,
    )
    dr.assert_no_hidden_tokens(prompt)
    print(f"prompt 长度: {len(prompt)}")
    raw = dr.call_llm(api_key, [{"role": "user", "content": prompt}], temperature=0.7)
    parsed = dr.parse_llm_proposal(raw, trial_seq=999, parent_trial="parent-seed42-formal")
    if not parsed.ok:
        print("提案解析失败:", parsed.errors)
        continue
    t = parsed.trial_record
    print("action_id:", t["action_id"], "| is_free:", t["is_free_proposal"])
    print("hypothesis:", t["hypothesis"][:120])
    print("patch_plan:", t["patch_plan"][:150])
    target = dr.AXIS_FILES[axis]
    src, errs = dr.gen_code_edit(api_key, axis, t, target.read_text(encoding="utf-8"))
    if src is None:
        print("代码生成失败:", errs)
        continue
    print(f"代码生成 OK（{len(src)} 字符，编译/签名/token 检查全过）")
    cfg = dr.load_config(dr.CONTRACT / "axis_lock_v1_draft.json", dr.AEROWF_REPO, dr.DOWNSTREAM)
    dec = dr.check([str(target)], axis, cfg)
    print("axis_lock:", "放行" if dec.allowed else f"拒绝 {dec.violations}")
print("\nPREFLIGHT_DONE")
