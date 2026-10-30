"""Deployment table view: medians over 10 steady batches; decode tok/s pooled.

    python -m precision.summarize_timing runs/precision/timing
"""
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
ARMS = ['base.bfloat16', 'md.bfloat16', 'md.float32', 'mlp.float32', 'mlp.bfloat16', 'res.float32', 'res.bfloat16']
out = {}
for arm in ARMS:
    p = root / arm / 'summary.json'
    if not p.exists(): continue
    s = json.loads(p.read_text()); assert s['status'] == 'complete'
    h = json.loads((root / arm / 'header.json').read_text())
    out[arm] = dict(memory_payload_bytes=(h['weights']['memory'] or {}).get('parameter_payload_bytes'), gpu=h['hardware']['gpu_name'], conditions={})
    for c in s['conditions']:
        m = c['metrics']
        out[arm]['conditions'][c['condition']] = dict(ttft_ms=1000 * m['ttft_seconds']['median'], total_ms=1000 * m['elapsed_seconds']['median'],
            decode_tok_s=m['pooled_decode_tokens_per_second'], peak_gib=m['peak_allocated_bytes']['median'] / 2**30)
base = out.get('base.bfloat16')
for arm, v in out.items():
    for c, m in v['conditions'].items():
        if base and c in base['conditions']:
            b = base['conditions'][c]; m['total_overhead_vs_base_pct'] = 100 * (m['total_ms'] / b['total_ms'] - 1); m['peak_over_base_gib'] = m['peak_gib'] - b['peak_gib']
(root / 'TIMING_SUMMARY.json').write_text(json.dumps(out, indent=1))
print('| arm | cond | TTFT ms | Total ms | Decode tok/s | Peak GiB | +Total% | +Peak GiB |')
for arm, v in out.items():
    for c in ('short_qa.b16.fixed', 'short_qa.b1.fixed'):
        m = v['conditions'].get(c)
        if m: print(f"| {arm} | {c} | {m['ttft_ms']:.2f} | {m['total_ms']:.2f} | {m['decode_tok_s']:.2f} | {m['peak_gib']:.3f} | {m.get('total_overhead_vs_base_pct', 0):.1f} | {m.get('peak_over_base_gib', 0):.3f} |")
