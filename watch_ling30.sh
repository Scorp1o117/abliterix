#!/bin/bash
# watch_ling30.sh — one-shot status for Ling abliteration runs
# Usage:
#   ./watch_ling30.sh              # auto: newest v3 log + v3 checkpoint
#   ./watch_ling30.sh v3           # v3 SRA probe
#   ./watch_ling30.sh v2           # v2 Laguna
#   watch -n 30 ./watch_ling30.sh  # refresh every 30s
set -euo pipefail
cd "$(dirname "$0")"

VER="${1:-v60}"
case "$VER" in
  v60|prefixretry)
    CKPT=checkpoints_ling30_flash_v60_prefix_retry_formal
    LOG_GLOB='logs/ling30_v60*.log'
    BUDGET=2
    ;;
  v59|steeredpeel)
    CKPT=checkpoints_ling30_flash_v59_steered_peel_formal
    LOG_GLOB='logs/ling30_v59*.log'
    BUDGET=2
    ;;
  v58|steereddump)
    CKPT=checkpoints_ling30_flash_v58_steered_residual_dump
    LOG_GLOB='logs/ling30_v58*.log'
    BUDGET=1
    ;;
  v57|harmfulness)
    CKPT=checkpoints_ling30_flash_v57_harmfulness_counterfactual
    LOG_GLOB='logs/ling30_v57*.log'
    BUDGET=2
    ;;
  v56|strengthfallback)
    CKPT=checkpoints_ling30_flash_v56_formal_strength_fallback_ab
    LOG_GLOB='logs/ling30_v56*.log'
    BUDGET=2
    ;;
  v55|signedgatefeature)
    CKPT=checkpoints_ling30_flash_v55_signed_gate_feature_probe
    LOG_GLOB='logs/ling30_v55*.log'
    BUDGET=1
    ;;
  v54|strengthfeature)
    CKPT=checkpoints_ling30_flash_v54_strength_feature_probe
    LOG_GLOB='logs/ling30_v54*.log'
    BUDGET=1
    ;;
  v53|projectiongeometry)
    CKPT=checkpoints_ling30_flash_v53_projection_geometry_calibration
    LOG_GLOB='logs/ling30_v53*.log'
    BUDGET=4
    ;;
  v52|overrotationphase)
    CKPT=checkpoints_ling30_flash_v52_overrotation_phase_calibration
    LOG_GLOB='logs/ling30_v52*.log'
    BUDGET=3
    ;;
  v51|overrotation)
    CKPT=checkpoints_ling30_flash_v51_overrotation_calibration
    LOG_GLOB='logs/ling30_v51*.log'
    BUDGET=4
    ;;
  v50|negativeformal)
    CKPT=checkpoints_ling30_flash_v50_negative_failure_formal_repro
    LOG_GLOB='logs/ling30_v50*.log'
    BUDGET=2
    ;;
  v49|negativefailure)
    CKPT=checkpoints_ling30_flash_v49_negative_failure_variants
    LOG_GLOB='logs/ling30_v49*.log'
    BUDGET=5
    ;;
  v48|renormprimary)
    CKPT=checkpoints_ling30_flash_v48_renormalized_primary_repro
    LOG_GLOB='logs/ling30_v48*.log'
    BUDGET=2
    ;;
  v47|failurevariants)
    CKPT=checkpoints_ling30_flash_v47_failure_conditioned_variants
    LOG_GLOB='logs/ling30_v47*.log'
    BUDGET=5
    ;;
  v46|traincounterfactual)
    CKPT=checkpoints_ling30_flash_v46_train_counterfactual_directions
    LOG_GLOB='logs/ling30_v46*.log'
    BUDGET=2
    ;;
  v45|legacyrouter)
    CKPT=checkpoints_ling30_flash_v45_legacy_primary_router
    LOG_GLOB='logs/ling30_v45*.log'
    BUDGET=1
    ;;
  v44|directionrouter)
    CKPT=checkpoints_ling30_flash_v44_direction_router_probe
    LOG_GLOB='logs/ling30_v44*.log'
    BUDGET=1
    ;;
  v43|legacyrank1)
    CKPT=checkpoints_ling30_flash_v43_legacy_rank1_frozen_gate
    LOG_GLOB='logs/ling30_v43*.log'
    BUDGET=1
    ;;
  v42a|frozenrank1)
    CKPT=checkpoints_ling30_flash_v42a_frozen_gate_rank1
    LOG_GLOB='logs/ling30_v42a*.log'
    BUDGET=1
    ;;
  v42b|frozenrank2)
    CKPT=checkpoints_ling30_flash_v42b_frozen_gate_rank2
    LOG_GLOB='logs/ling30_v42b*.log'
    BUDGET=1
    ;;
  v41|rank2)
    CKPT=checkpoints_ling30_flash_v41_rank2_subspace_probe
    LOG_GLOB='logs/ling30_v41*.log'
    BUDGET=1
    ;;
  v40|gateattribution)
    CKPT=checkpoints_ling30_flash_v40_gate_attribution
    LOG_GLOB='logs/ling30_v40*.log'
    BUDGET=1
    ;;
  v39|outcomes)
    CKPT=checkpoints_ling30_flash_v39_outcome_map
    LOG_GLOB='logs/ling30_v39*.log'
    BUDGET=1
    ;;
  v38|flatdepth)
    CKPT=checkpoints_ling30_flash_v38_flat_full_depth_probe
    LOG_GLOB='logs/ling30_v38*.log'
    BUDGET=1
    ;;
  v37|bidirectional)
    CKPT=checkpoints_ling30_flash_v37_bidirectional_gate_probe
    LOG_GLOB='logs/ling30_v37*.log'
    BUDGET=1
    ;;
  v36|matchedcanonical)
    CKPT=checkpoints_ling30_flash_v36_matched_canonical_full
    LOG_GLOB='logs/ling30_v36*.log'
    BUDGET=1
    ;;
  v35|canonicalfull)
    CKPT=checkpoints_ling30_flash_v35_canonical_full
    LOG_GLOB='logs/ling30_v35*.log'
    BUDGET=1
    ;;
  v34|canonicalbatching)
    CKPT=checkpoints_ling30_flash_v34_canonical_batching
    LOG_GLOB='logs/ling30_v34*.log'
    BUDGET=1
    ;;
  v33|batchinvariance)
    CKPT=checkpoints_ling30_flash_v33_batch_invariance
    LOG_GLOB='logs/ling30_v33*.log'
    BUDGET=1
    ;;
  v32|globalprofiles)
    CKPT=checkpoints_ling30_flash_v32_global_profile_search
    LOG_GLOB='logs/ling30_v32*.log'
    BUDGET=8
    ;;
  v31|globalstrongfull)
    CKPT=checkpoints_ling30_flash_v31_global_strong_full
    LOG_GLOB='logs/ling30_v31*.log'
    BUDGET=1
    ;;
  v30|globalstrength)
    CKPT=checkpoints_ling30_flash_v30_global_strength
    LOG_GLOB='logs/ling30_v30*.log'
    BUDGET=3
    ;;
  v29|globalfull)
    CKPT=checkpoints_ling30_flash_v29_global_prompt_full
    LOG_GLOB='logs/ling30_v29*.log'
    BUDGET=1
    ;;
  v28|globalmaxgap)
    CKPT=checkpoints_ling30_flash_v28_global_prompt_maxgap
    LOG_GLOB='logs/ling30_v28*.log'
    BUDGET=1
    ;;
  v27|globalobserver)
    CKPT=checkpoints_ling30_flash_v27_global_prompt_observer
    LOG_GLOB='logs/ling30_v27*.log'
    BUDGET=1
    ;;
  v26|globalauto)
    CKPT=checkpoints_ling30_flash_v26_global_prompt_auto
    LOG_GLOB='logs/ling30_v26*.log'
    BUDGET=1
    ;;
  v25|globalgate)
    CKPT=checkpoints_ling30_flash_v25_global_prompt_gate
    LOG_GLOB='logs/ling30_v25*.log'
    BUDGET=1
    ;;
  v24|generated400)
    CKPT=checkpoints_ling30_flash_v24_generated_latch_400
    LOG_GLOB='logs/ling30_v24*.log'
    BUDGET=1
    ;;
  v23|generatedlatch)
    CKPT=checkpoints_ling30_flash_v23_generated_prompt_latch
    LOG_GLOB='logs/ling30_v23*.log'
    BUDGET=1
    ;;
  v22|generatedgate)
    CKPT=checkpoints_ling30_flash_v22_generated_gate
    LOG_GLOB='logs/ling30_v22*.log'
    BUDGET=1
    ;;
  v21|trajectorylow)
    CKPT=checkpoints_ling30_flash_v21_trajectory_gate_low
    LOG_GLOB='logs/ling30_v21*.log'
    BUDGET=1
    ;;
  v20|trajectorygate)
    CKPT=checkpoints_ling30_flash_v20_trajectory_gate
    LOG_GLOB='logs/ling30_v20*.log'
    BUDGET=1
    ;;
  v19|promptlow)
    CKPT=checkpoints_ling30_flash_v19_prompt_gate_low
    LOG_GLOB='logs/ling30_v19*.log'
    BUDGET=1
    ;;
  v18|promptgate)
    CKPT=checkpoints_ling30_flash_v18_prompt_gate
    LOG_GLOB='logs/ling30_v18*.log'
    BUDGET=1
    ;;
  v17|gatefull)
    CKPT=checkpoints_ling30_flash_v17_concept_gate_full
    LOG_GLOB='logs/ling30_v17*.log'
    BUDGET=1
    ;;
  v16|gate)
    CKPT=checkpoints_ling30_flash_v16_concept_gate
    LOG_GLOB='logs/ling30_v16*.log'
    BUDGET=3
    ;;
  v15|postattn)
    CKPT=checkpoints_ling30_flash_v15_post_attn
    LOG_GLOB='logs/ling30_v15*.log'
    BUDGET=1
    ;;
  v14|sites)
    CKPT=checkpoints_ling30_flash_v14_site_sweep
    LOG_GLOB='logs/ling30_v14*.log'
    BUDGET=4
    ;;
  v13|adaptive)
    CKPT=checkpoints_ling30_flash_v13_adaptive_runtime
    LOG_GLOB='logs/ling30_v13*.log'
    BUDGET=18
    ;;
  v12|response)
    CKPT=checkpoints_ling30_flash_v12_response_pair
    LOG_GLOB='logs/ling30_v12*.log'
    BUDGET=12
    ;;
  v11|paired)
    CKPT=checkpoints_ling30_flash_v11_paired
    LOG_GLOB='logs/ling30_v11*.log'
    BUDGET=12
    ;;
  v9|optimal)
    CKPT=checkpoints_ling30_flash_v9_optimal
    LOG_GLOB='logs/ling30_v9_*.log'
    BUDGET=60
    ;;
  v8|default)
    CKPT=checkpoints_ling30_flash_v8_default
    LOG_GLOB='logs/ling30_v8_*.log'
    BUDGET=40
    ;;
  v7|cliff)
    CKPT=checkpoints_ling30_flash_v7_cliff
    LOG_GLOB='logs/ling30_v7_*.log'
    BUDGET=10
    ;;
  v6|stack)
    CKPT=checkpoints_ling30_flash_v6_stack
    LOG_GLOB='logs/ling30_v6_*.log'
    BUDGET=80
    ;;
  v5|lowref)
    CKPT=checkpoints_ling30_flash_v5_lowref
    LOG_GLOB='logs/ling30_v5_*.log'
    BUDGET=20
    ;;
  v4|ornith)
    CKPT=checkpoints_ling30_flash_v4_ornith
    LOG_GLOB='logs/ling30_v4_*.log'
    BUDGET=10
    ;;
  b16)
    CKPT=checkpoints_ling30_flash_v4_b16
    LOG_GLOB='logs/ling30_v4_*.log'
    BUDGET=12
    ;;
  v3|sra)
    CKPT=checkpoints_ling30_flash_v3_sra
    LOG_GLOB='logs/ling30_v3_*.log'
    BUDGET=10
    ;;
  v2)
    CKPT=checkpoints_ling30_flash_v2
    LOG_GLOB='logs/ling30_v2_*.log'
    BUDGET=40
    ;;
  v1|debug)
    CKPT=checkpoints_ling30_flash_debug
    LOG_GLOB='logs/ling30_*.log'
    BUDGET=20
    ;;
  *)
    echo "usage: $0 [v60|v59|v58|v57|v56|v55|v54|v53|v52|v51|v50|v49|v48|v47|v46|v45|v44|v43|v42a|v42b|v41|v40|v39|v38|v37|v36|v35|v34|v33|v32|v31|v30|v29|v28|v27|v26|v25|v24|v23|v22|v21|v20|v19|v18|v17|v16|v15|v14|v13|v12|v11|v9|v8|v7|v6|v5|v4|b16|v3|v2|v1]"; exit 1
    ;;
esac

LOG=$(ls -t $LOG_GLOB 2>/dev/null | head -1 || true)
J=$(ls "$CKPT"/*flash.jsonl 2>/dev/null | head -1 || true)

echo "=== Ling $VER status $(date '+%F %T') ==="
# process
PIDS=$(pgrep -x abliterix 2>/dev/null || true)
if [ -n "${PIDS:-}" ]; then
  ps -o pid,etime,pcpu,pmem,cmd -p $PIDS 2>/dev/null || true
else
  echo "abliterix: not running"
fi

echo
echo "--- log: ${LOG:-none} ---"
if [ -n "${LOG:-}" ] && [ -f "$LOG" ]; then
  # milestones (no progress-bar spam)
  grep -E 'Trying dtype|Ok \(quantized|LoRA adapters|Steering|Baseline|Vector method|concept scorers|Concept gate|concept gate|Trajectory|trajectory|Validation KL|Generation health|Thinking leak|SRA|Cliff-head|cliff-head|Ablated|Running trial|Refusal prescreen|Refusals|exceeds prune|refusals=|Objective|Error|Traceback|Trial [0-9]+ finished|Pareto|Best trial' "$LOG" 2>/dev/null | tail -25 || true
  echo
  echo "(last line)"
  tail -c 400 "$LOG" | tr '\r' '\n' | tail -3
fi

echo
echo "--- trials (budget $BUDGET) ---"
if [ -n "${J:-}" ] && [ -f "$J" ]; then
  python3 - "$J" "$BUDGET" <<'PY'
import json, math, sys
from collections import defaultdict
path, budget = sys.argv[1], int(sys.argv[2])
trials = defaultdict(lambda: {"params": {}, "ua": {}})
for line in open(path):
    line = line.strip()
    if not line:
        continue
    r = json.loads(line)
    tid = r.get("trial_id")
    if tid is None:
        continue
    if r.get("op_code") == 5:
        trials[tid]["params"][r["param_name"]] = r.get("param_value_internal")
    if r.get("op_code") == 6 and r.get("values") is not None:
        trials[tid]["values"] = r["values"]
    if r.get("op_code") == 8 and isinstance(r.get("user_attr"), dict):
        trials[tid]["ua"].update(r["user_attr"])

done = ok = inf = run = 0
print(f"{'tid':>3} {'st':>3} {'pr':>4} {'KL':>8} {'ref%':>6}  notes")
best = None
for tid in sorted(trials):
    t = trials[tid]
    ua, v = t["ua"], t.get("values")
    pr = ua.get("prescreen_refusals")
    kl_a = ua.get("kl_divergence")
    o = t["params"].get("attn.o_proj.max_weight")
    d = t["params"].get("mlp.down_proj.max_weight")
    note = []
    if o is not None:
        note.append(f"o={o:.2f}")
    if d is not None:
        note.append(f"d={d:.2f}")
    var = ua.get("steering_variant")
    if var:
        note.append(str(var))
    site = ua.get("runtime_hook_site") or t["params"].get("runtime_hook_site")
    if site:
        note.append(f"site={site}")
    gate_thr = t["params"].get("concept_gate_threshold")
    gate_rate = ua.get("concept_gate_active_rate")
    if gate_thr is not None:
        note.append(f"gate>={gate_thr:.2f}")
    if gate_rate is not None:
        note.append(f"active={gate_rate*100:.1f}%")
    if v is None:
        run += 1
        st, kl_s, ref_s = "RUN", "-", "-"
    else:
        done += 1
        a, b = float(v[0]), float(v[1])
        if math.isfinite(a) and math.isfinite(b):
            ok += 1
            st = "OK"
            kl_s = f"{a:.4f}"
            ref_s = f"{b*100:.1f}"
            hit = a <= 0.05 and b <= 0.10
            near = a <= 0.15 and b <= 0.20
            if hit:
                note.append("HIT")
            elif near:
                note.append("NEAR")
            if best is None or (a, b) < (best[0], best[1]):
                best = (a, b, tid)
        else:
            inf += 1
            st = "INF"
            kl_s = f"{kl_a:.4f}" if isinstance(kl_a, (int, float)) else "inf"
            ref_s = "-"
            if pr is not None:
                note.append(f"pr={pr}")
    print(f"{tid:3d} {st:>3} {str(pr) if pr is not None else '-':>4} {kl_s:>8} {ref_s:>6}  {' '.join(note)}")

print(f"\ndone={done}/{budget}  OK={ok}  INF/prune={inf}  runningish={run}")
if best:
    print(f"best finite: t{best[2]} KL={best[0]:.4f} ref={best[1]*100:.1f}%  (goal KL≤0.05 ref≤10%)")
else:
    print("best finite: none yet")
PY
else
  echo "journal: not created yet ($CKPT)"
fi
