"""Aggregate test-set EM, F1 and reference-answer PPL into the knowledge-QA table.

Open-domain QA averages NQ, WebQuestions, TriviaQA and PopQA equally; Multi-hop
QA is HotpotQA; the overall mean weights the five datasets equally. Category PPL
averages dataset perplexities. Each method is read
from its qa.run_grid test directory at its selected coefficient (Base and LoRA
use coefficient 0; BM25-RAG outputs use the same file layout).

    python -m qa.evaluate --split-root data/qa/test/frozen \
        --method Base=runs/base/test --method ResMem=runs/resmem/test:runs/resmem/selected.json ...
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from qa.common import DATASETS, cell_path, load_rows, perplexity
from qa.select_coefficients import score_cell

OPEN_DOMAIN = ("nq", "webq", "triviaqa", "popqa")
MULTI_HOP = ("hotpotqa",)


def evaluate_method(split_root: Path, grid: Path, selected: Path | None) -> dict:
    coefficients = json.loads(selected.read_text())["coefficients"] if selected else {ds: 0.0 for ds in DATASETS}
    result = {}
    for ds in DATASETS:
        rows = load_rows(split_root / f"{ds}.jsonl")
        coefficient = float(coefficients[ds])
        metrics = score_cell(rows, cell_path(grid, ds, coefficient, "predictions"))
        nll = cell_path(grid, ds, coefficient, "nll")
        if nll.exists():
            metrics["ppl"] = perplexity(load_rows(nll))
        result[ds] = {"coefficient": coefficient, "questions": len(rows), **metrics}
    for name, members in (("open_domain", OPEN_DOMAIN), ("multi_hop", MULTI_HOP), ("mean", DATASETS)):
        keys = [k for k in ("em", "f1", "ppl") if all(k in result[ds] for ds in members)]
        result[name] = {k: sum(result[ds][k] for ds in members) / len(members) for k in keys}
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split-root", required=True, type=Path, help="test <dataset>.jsonl files")
    parser.add_argument("--method", action="append", required=True,
                        help="NAME=TEST_GRID_DIR[:SELECTED_JSON]; repeat for every row of the table")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    table = {}
    for spec in args.method:
        name, _, paths = spec.partition("=")
        grid, _, selected = paths.partition(":")
        table[name] = evaluate_method(args.split_root, Path(grid), Path(selected) if selected else None)
    print(f"{'method':<16}{'OD EM':>8}{'OD F1':>8}{'OD PPL':>8}{'MH EM':>8}{'MH F1':>8}{'MH PPL':>8}{'EM':>8}{'F1':>8}")
    for name, result in table.items():
        cells = [100 * result[c].get(k, float("nan")) if k != "ppl" else result[c].get(k, float("nan"))
                 for c in ("open_domain", "multi_hop") for k in ("em", "f1", "ppl")]
        cells += [100 * result["mean"][k] for k in ("em", "f1")]
        print(f"{name:<16}" + "".join(f"{v:>8.2f}" for v in cells))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(table, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
