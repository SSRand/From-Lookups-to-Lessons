"""One immutable winner, chosen before any audit or final-evaluation outputs.

    python -m gate.freeze --development runs/gate/development-final --pairs runs/gate/fitdev --output runs/gate
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

HERE = Path(__file__).resolve().parent
p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
p.add_argument('--development', type=Path, required=True); p.add_argument('--pairs', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
args = p.parse_args()
root = args.development
leaderboard = json.loads((root/'LEADERBOARD.json').read_text())
candidates = json.loads((root/'CANDIDATES.json').read_text())
best = leaderboard[0]['name']
path = args.output/'FROZEN.json'
assert not path.exists(), 'winner already frozen; do not reselect'
parent_request = json.loads((args.pairs/'REQUEST.json').read_text())
obj = {'frozen_at': time.time(), 'candidate_name': best, 'candidate': candidates[best],
       'development_score': leaderboard[0],
       'leaderboard_sha256': hashlib.sha256((root/'LEADERBOARD.json').read_bytes()).hexdigest(),
       'candidate_file_sha256': hashlib.sha256((root/'CANDIDATES.json').read_bytes()).hexdigest(),
       'inference_code_sha256': {p: hashlib.sha256((HERE/p).read_bytes()).hexdigest() for p in ['runner.py', 'gates.py']},
       'protocol_identity': {k: parent_request[k] for k in ['data_sha', 'lambdas', 'protocol', 'batch_size']},
       'selection_rule': 'Two-domain dev macro EM, then macro token F1, then serialized policy complexity; no audit/test selection.',
       'deployment_rule': 'Report this winner even if it fails. Retain fixed RES as supported incumbent unless both audit and five-set macro EM improve; report confidence intervals and no guarantee of per-domain superiority.'}
args.output.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(obj, indent=2))
(args.output/'FINAL_CANDIDATE.json').write_text(json.dumps({best: candidates[best]}, indent=2))
print(json.dumps(obj, indent=2))
