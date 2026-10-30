"""Select one coefficient per dataset by the highest EM on the validation set.

Ties are broken by F1 and then by the smaller coefficient. The selected
coefficient is then used unchanged for EM, F1 and PPL on the test set.

Selection starts from a coarse grid and is refined twice with the midpoints
between the current winner and its immediate tested neighbours; each call also
writes those midpoints to ``--refine-output`` for the next qa.run_grid pass.

    python -m qa.select_coefficients --split-root data/qa/validation/frozen \
        --grid runs/resmem/validation --output runs/resmem/selected.json --refine-output runs/resmem/refine.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from qa.common import DATASETS, load_rows
from resmem.scoring import summarize_predictions


def score_cell(rows: list[dict], path: Path) -> dict[str, float]:
    metrics = summarize_predictions(rows, load_rows(path), protocol="closed_book")["metrics"]
    return {"em": metrics["exact_match"], "f1": metrics["token_f1"]}


def select(candidates: list[dict]) -> dict:
    return max(candidates, key=lambda x: (x["em"], x["f1"], -x["coefficient"]))


def midpoints(candidates: list[dict], winner: float) -> list[float]:
    tested = sorted(c["coefficient"] for c in candidates)
    index = tested.index(winner)
    neighbours = tested[max(0, index - 1) : index] + tested[index + 1 : index + 2]
    return sorted({round((winner + n) / 2, 10) for n in neighbours} - set(tested))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split-root", required=True, type=Path, help="validation <dataset>.jsonl files")
    parser.add_argument("--grid", required=True, type=Path, help="qa.run_grid output directory for the validation split")
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--refine-output", type=Path, help="write the next refinement grid (per-dataset midpoints)")
    args = parser.parse_args(argv)

    report = {"rule": "validation EM, then validation F1, then smaller coefficient", "coefficients": {}, "candidates": {}}
    for ds in args.datasets:
        rows = load_rows(args.split_root / f"{ds}.jsonl")
        candidates = []
        for path in sorted((args.grid / ds).glob("*.predictions.jsonl")):
            coefficient = float(path.name.split(".predictions.jsonl")[0])
            candidates.append({"coefficient": coefficient, **score_cell(rows, path)})
        if not candidates:
            raise FileNotFoundError(f"no validation predictions for {ds} under {args.grid}")
        report["candidates"][ds] = sorted(candidates, key=lambda x: x["coefficient"])
        report["coefficients"][ds] = select(candidates)["coefficient"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if args.refine_output:
        refine = {ds: midpoints(report["candidates"][ds], report["coefficients"][ds]) for ds in args.datasets}
        args.refine_output.write_text(json.dumps({"coefficients": refine}, indent=2) + "\n")
    print(json.dumps(report["coefficients"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
