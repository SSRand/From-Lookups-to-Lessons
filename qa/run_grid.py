"""Generate answers (and optionally score reference answers) over a coefficient grid.

Work items are (dataset, coefficient) cells. Launch one process per GPU with
``--shard i --num-shards n``; each cell is written once and skipped on rerun.

    # validation grid for coefficient selection (then refine with --selected refine.json)
    python -m qa.run_grid --split-root data/qa/validation/frozen --method resmem \
        --checkpoint CKPT --base-model mistralai/Mistral-7B-v0.3 \
        --coefficients 0 .05 .1 .15 .2 .25 .3 .4 .5 .6 .8 1 1.25 1.5 2 --output runs/resmem/validation

    # test set at the coefficients chosen on validation (EM, F1 and reference-answer NLL)
    python -m qa.run_grid --split-root data/qa/test/frozen --method resmem --checkpoint CKPT \
        --selected runs/resmem/selected.json --score-answers --output runs/resmem/test
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from qa.common import DATASETS, cell_path, load_rows, write_jsonl
from qa.runtime import METHODS, QARuntime


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--split-root", type=Path, help="directory with frozen <dataset>.jsonl files")
    source.add_argument("--rows", type=Path, help="one JSONL file whose rows carry a 'dataset' field (e.g. a development pool)")
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--checkpoint", help="memory checkpoint, Memory Decoder checkpoint, or LoRA adapter file")
    parser.add_argument("--base-model", default="mistralai/Mistral-7B-v0.3")
    grid = parser.add_mutually_exclusive_group(required=True)
    grid.add_argument("--coefficients", nargs="+", type=float, help="one grid shared by all datasets")
    grid.add_argument("--selected", type=Path, help="per-dataset coefficients: selected.json (one value) or refine.json (a list)")
    parser.add_argument("--lora-strength", type=float, default=0.25)
    parser.add_argument("--score-answers", action="store_true", help="also write reference-answer token NLL")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args(argv)

    if args.rows is not None:
        pool = load_rows(args.rows)
        args.datasets = [ds for ds in args.datasets if any(r["dataset"] == ds for r in pool)]
    if args.selected is not None:
        chosen = json.loads(args.selected.read_text())["coefficients"]
        grids = {ds: [float(v) for v in (chosen[ds] if isinstance(chosen[ds], list) else [chosen[ds]])] for ds in args.datasets}
    else:
        grids = {ds: sorted(set(args.coefficients)) for ds in args.datasets}
    if args.method in {"base", "lora"}:
        grids = {ds: [0.0] for ds in args.datasets}
    cells = [(ds, c) for ds in args.datasets for c in grids[ds]]
    mine = [cell for index, cell in enumerate(cells) if index % args.num_shards == args.shard]
    pending = [
        (ds, c) for ds, c in mine
        if not cell_path(args.output, ds, c, "predictions").exists()
        or (args.score_answers and not cell_path(args.output, ds, c, "nll").exists())
    ]
    if not pending:
        return 0

    runtime = QARuntime(args.base_model)
    runtime.set_method(args.method, args.checkpoint, lora_strength=args.lora_strength)
    if args.rows is not None:
        data = {ds: [r for r in pool if r["dataset"] == ds] for ds in {ds for ds, _ in pending}}
    else:
        data = {ds: load_rows(args.split_root / f"{ds}.jsonl") for ds in {ds for ds, _ in pending}}
    for ds, coefficient in pending:
        rows = data[ds]
        predictions = cell_path(args.output, ds, coefficient, "predictions")
        if not predictions.exists():
            write_jsonl(predictions, runtime.generate(rows, coefficient))
        nll = cell_path(args.output, ds, coefficient, "nll")
        if args.score_answers and not nll.exists():
            write_jsonl(nll, runtime.score_answers(rows, [coefficient])[float(coefficient)])
        print(json.dumps({"dataset": ds, "coefficient": coefficient, "rows": len(rows)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
