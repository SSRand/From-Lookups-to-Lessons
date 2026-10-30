"""Assemble the directory read by memory_decoder.aggregate.

  python -m shared_teacher_ablation.memory_decoder.stage_inputs --population-root POP --md-root MDOUT \\
      --eval-test-root EVAL/test [--eval-test-root ...] --output AGG
Layout: AGG/test.jsonl (population test rows), AGG/md/s{seed}.jsonl and AGG/md/base{seed}.jsonl
(Memory Decoder test rows and Base shards from evaluate.py), AGG/ctrl/{objective}-s{seed}.jsonl
(native-readout test rows of both shared-teacher objectives, models/<cell>/native_selected).
"""
import argparse,shutil
from pathlib import Path
SEEDS=(42,123,456,789,2026,2027,2028,2029)
OBJECTIVES=('standalone_memory','joint_base_anchored')

def main(a):
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False);(out/'md').mkdir();(out/'ctrl').mkdir()
    shutil.copyfile(Path(a.population_root)/'test.jsonl',out/'test.jsonl')
    md=Path(a.md_root)
    for s in SEEDS:
        if not (md/f's{s}'/'COMPLETE.json').exists():raise ValueError(f'incomplete Memory Decoder seed {s}')
        shutil.copyfile(md/f's{s}'/'test.jsonl',out/'md'/f's{s}.jsonl');shutil.copyfile(md/f's{s}'/'base_shard.jsonl',out/'md'/f'base{s}.jsonl')
    for o in OBJECTIVES:
        for s in SEEDS:
            cell=f'{o}-s{s}';found=[Path(r)/'models'/cell/'native_selected' for r in a.eval_test_root if (Path(r)/'models'/cell/'native_selected'/'COMPLETE.json').exists()]
            if len(found)!=1:raise ValueError(f'exactly one completed native test capture required for {cell}')
            shutil.copyfile(found[0]/'predictions.jsonl',out/'ctrl'/f'{cell}.jsonl')

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    for n in ('population-root','md-root','output'):ap.add_argument('--'+n,required=True)
    ap.add_argument('--eval-test-root',action='append',required=True)
    main(ap.parse_args())
