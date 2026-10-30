"""Write the ordered cache manifest consumed by train.pretrain (--cache-manifest).

The manifest fixes the file order of the teacher cache (sorted *.npz under --cache), the per-file token
counts that set the steps per epoch, and per-file size/mtime/sha256 for integrity checks.
Usage: python -m train.cache_manifest --cache <teacher cache dir> --out <cache_manifest.json>
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np

from train.checkpointing import atomic_write_json


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_cache_manifest(cache_root, *, with_sha256=True, builder_manifest=None, topk=64):
    cache_root = Path(cache_root).resolve(strict=False)
    files = sorted(cache_root.glob("*.npz"))
    if not files:
        raise ValueError(f"no *.npz cache files under {cache_root}")
    span_digest = hashlib.sha256() if with_sha256 else None
    entries = []
    span_count = 0
    cache_row_count = 0
    token_count = 0
    for path in files:
        with np.load(path, allow_pickle=False) as payload:
            span_lens = payload["span_lens"].astype(np.int64)
            expected_rows = int(np.clip(span_lens - 1, 0, None).sum())
            if with_sha256:
                tokens = payload["teacher_tok"]
                probabilities = payload["teacher_prob"]
                if not (
                    tokens.ndim == 2
                    and tokens.shape[1] == topk
                    and tokens.dtype == np.int32
                    and probabilities.shape == tokens.shape
                    and probabilities.dtype == np.float16
                ):
                    raise ValueError(f"unexpected cache schema in {path}")
                if int(tokens.shape[0]) != expected_rows:
                    raise ValueError(f"cache rows do not match span lengths in {path}")
                span_ids = payload["span_ids"].astype(np.int32)
                rows = np.concatenate([span_lens.astype(np.int32)[:, None], span_ids], axis=1)
                mask = np.arange(rows.shape[1])[None, :] <= span_lens[:, None]
                span_digest.update(rows[mask].tobytes())
        stat = path.stat()
        entries.append(
            {
                "relative_path": path.relative_to(cache_root).as_posix(),
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
                "sha256": _sha256_file(path) if with_sha256 else None,
                "span_count": int(span_lens.shape[0]),
                "token_count": expected_rows,
                "cache_rows": expected_rows,
            }
        )
        span_count += int(span_lens.shape[0])
        cache_row_count += expected_rows
        token_count += expected_rows
    payload = {
        "schema_version": "cache-identity-v1",
        "status": "complete",
        "cache_root": str(cache_root),
        "ordered_cache_entries": entries,
        "span_count": span_count,
        "cache_row_count": cache_row_count,
        "token_count": token_count,
        "span_order_sha256": span_digest.hexdigest() if span_digest is not None else None,
    }
    if builder_manifest:
        builder_path = Path(builder_manifest).resolve(strict=False)
        payload["builder_manifest"] = {"path": str(builder_path), "sha256": _sha256_file(builder_path)}
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True, help="directory of teacher.forward_teacher outputs (*.npz)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--builder-manifest", default=None, help="optional file recording how the cache was built")
    parser.add_argument("--topk", type=int, default=64)
    args = parser.parse_args()
    payload = build_cache_manifest(args.cache, builder_manifest=args.builder_manifest, topk=args.topk)
    atomic_write_json(args.out, payload)
    print(
        f"wrote {args.out}: files={len(payload['ordered_cache_entries'])} spans={payload['span_count']} "
        f"tokens={payload['token_count']} sha256={_sha256_file(Path(args.out))}",
        flush=True,
    )


if __name__ == "__main__":
    main()
