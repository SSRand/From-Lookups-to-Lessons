# From Lookups to Lessons: Turning Retrieved Knowledge into Parametric Memory

Changjiang Han, Ye Yuan, Bowei He, Weixu Zhang, Haolun Wu, Xue Liu, Jikun Kang

MBZUAI, McGill University, Mila – Quebec AI Institute, Salesforce Research

[Paper (arXiv)](#citation) | [Model weights (Hugging Face)](https://huggingface.co/RandHan/ResMem-Mistral-7B-v0.3)

ResMem trains a feed-forward memory on top of a frozen Mistral-7B-v0.3 by residual self-distillation from a BM25-RAG teacher, and reads it out additively: softmax(z_B + γ s).

## Layout

| Directory | Contents | Paper |
| --- | --- | --- |
| `resmem/` | Memory architecture, readout rules, prompt, EM/F1 scoring, reference-answer NLL | all |
| `teacher/` | BM25 retrieval and teacher target caching | Sec. 3 |
| `train/` | ResMem trainer | Sec. 3 |
| `qa/` | Knowledge-QA data, inference, validation-set coefficient selection, aggregation (BM25-RAG under `qa/baselines/`) | Table 1 |
| `correction_harm/` | Correction/harm analysis | Figure 4 |
| `shared_teacher_ablation/` | Shared-teacher ablation | Table 2 |
| `precision/` | Precision and deployment cost | Appendix C |
| `gate/` | Tokenwise gate | Appendix D |

For the baselines (MLP Memory, Memory Decoder, LoRA, BM25-RAG) we provide evaluation code only.
Settings are recorded in [`docs/experimental_protocol.md`](docs/experimental_protocol.md).
Entry points are Python modules run from the repository root; each module's docstring gives its usage.

## Setup

```bash
pip install -r requirements.txt
```

## Pretrained memory

The knowledge-QA checkpoint used in Table 1 is released at [RandHan/ResMem-Mistral-7B-v0.3](https://huggingface.co/RandHan/ResMem-Mistral-7B-v0.3).
It is trained for one epoch on the December 2021 English Wikipedia and attaches to the frozen `mistralai/Mistral-7B-v0.3`.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from resmem.memory import capture_memory_input, load_memory

tok = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.3")
base = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.3", torch_dtype=torch.bfloat16).cuda().eval()
memory = load_memory("RandHan/ResMem-Mistral-7B-v0.3", base, "cuda")

store = {}
capture_memory_input(base, store)
ids = tok("Q: Who wrote Hamlet?\nA:", return_tensors="pt").input_ids.cuda()
with torch.no_grad():
    z_base = base(ids).logits[:, -1].float()
    s = memory(store["hidden_states"][:, -1].float())
    z = z_base + 0.4 * s  # gamma = 0.4
print(tok.decode(z.argmax(-1)))
```

Pass the Hugging Face id as `CKPT` in the commands below to reproduce Table 1.

## Training

```bash
# 1. BM25 index over pre-tokenized Wikipedia shards (train/*.arrow with input_ids/labels)
python -m teacher.build_bm25_index --data WIKI --out bm25

# 2. Retrieve the top-3 passages for each 64-token span and assemble (context + span) sequences (CPU)
python -m teacher.prepare_contexts --data WIKI --bm25 bm25 --out prepped --workers 32

# 3. Cache the top-64 teacher distributions of the frozen base reading the evidence (GPU)
python -m teacher.forward_teacher --prepped prepped --out teacher_cache --world 8

# 4. Train the memory on the frozen base
python -m train.cache_manifest --cache teacher_cache --out teacher_cache/manifest.json
python -m train.pretrain --cache teacher_cache --cache-manifest teacher_cache/manifest.json --out runs/resmem --world 8 --epochs 1
```

`WIKI` is the pre-tokenized December 2021 English Wikipedia, e.g. [Rubin-Wei/enwiki-dec2021-preprocessed-mistral](https://huggingface.co/datasets/Rubin-Wei/enwiki-dec2021-preprocessed-mistral).

## Knowledge QA (Table 1)

```bash
# Test and validation sets
python -m qa.data.build_test_sets --nq NQ_TEST --triviaqa TQA_TEST --webq WEBQ_TEST --hotpotqa HOTPOT_DEV --popqa POPQA --data-root data/qa/test
python -m qa.data.build_validation_sets --test-root data/qa/test/frozen --nq NQ_DEV --triviaqa TQA_DEV --webq WEBQ_DEV \
    --hotpotqa HOTPOT_TRAIN --popqa ENTITYQUESTIONS --output-root data/qa/validation

# Select the coefficient on validation EM, then evaluate on the test set
python -m qa.run_grid --split-root data/qa/validation/frozen --method resmem --checkpoint CKPT --coefficients GRID --output runs/resmem/validation
python -m qa.select_coefficients --split-root data/qa/validation/frozen --grid runs/resmem/validation --output runs/resmem/selected.json
python -m qa.run_grid --split-root data/qa/test/frozen --method resmem --checkpoint CKPT --selected runs/resmem/selected.json --score-answers --output runs/resmem/test
python -m qa.evaluate --split-root data/qa/test/frozen --method ResMem=runs/resmem/test:runs/resmem/selected.json
```

The same commands apply to `--method mlpmemory` and `--method memory_decoder`; `base` and `lora` take no coefficient.

## Citation

```bibtex
@article{han2026lookups,
  title   = {From Lookups to Lessons: Turning Retrieved Knowledge into Parametric Memory},
  author  = {Han, Changjiang and Yuan, Ye and He, Bowei and Zhang, Weixu and Wu, Haolun and Liu, Xue and Kang, Jikun},
  journal = {arXiv preprint},
  year    = {2026}
}
```
