"""Bind final runs to the immutable winner before launching any inference.

Arguments are forwarded to gate.runner; audit pairs use ``--phase collect
--splits audit``, final evaluations ``--phase evaluate`` with the frozen candidate.

    python -m gate.run_frozen --frozen runs/gate/FROZEN.json --phase evaluate --scope test --splits test \
        --candidates runs/gate/FINAL_CANDIDATE.json --test-root data/qa/test/frozen --calibration ... \
        --selected runs/resmem/selected.json --checkpoint CKPT --output runs/gate/test-winner
"""
import hashlib
import json
from pathlib import Path
import sys
import time
from gate import runner

HERE = Path(__file__).resolve().parent
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
argv = sys.argv[1:]
frozen = Path(argv[argv.index('--frozen')+1]); del argv[argv.index('--frozen'):argv.index('--frozen')+2]
args = runner.parser().parse_args(argv)
freeze = json.loads(frozen.read_text())
assert args.scope == 'test' or args.splits == 'audit'
assert args.phase in ['collect', 'evaluate']
if args.phase == 'evaluate':
    assert json.loads(args.candidates.read_text()) == {freeze['candidate_name']: freeze['candidate']}
else:
    assert args.scope == 'calibration' and args.splits == 'audit' and args.candidates is None
for name, digest in freeze['inference_code_sha256'].items(): assert sha(HERE/name) == digest
args.output.mkdir(parents=True, exist_ok=True)
binding = {'freeze_sha256': sha(frozen), 'code': freeze['inference_code_sha256'],
           'argv': argv, 'started_at': time.time(), 'frozen_at': freeze['frozen_at']}
assert binding['started_at'] >= binding['frozen_at']
target = args.output/'FREEZE_BINDING.json'
if target.exists():
    old = json.loads(target.read_text())
    for k in ['freeze_sha256', 'code', 'frozen_at']: assert old[k] == binding[k]
else:
    assert not (args.output/'REQUEST.json').exists() and not (args.output/'COMPLETION.json').exists()
    assert not any((args.output/'tasks').glob('*')), 'refuse retroactive freeze binding'
    target.write_text(json.dumps(binding, indent=2))
runner.main(args)
request = json.loads((args.output/'REQUEST.json').read_text())
for key, value in freeze['protocol_identity'].items(): assert request[key] == value
assert request['code'] == freeze['inference_code_sha256']
if (args.output/'COMPLETION.json').exists():
    assert json.loads((args.output/'COMPLETION.json').read_text())['identity'] == request['identity']
    print(json.dumps({'final_run_verified': str(args.output), 'freeze_sha256': binding['freeze_sha256']}), flush=True)
