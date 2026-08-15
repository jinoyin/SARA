# ULTRA Self-Evolution with Structural KG Augmentation

This package contains the minimal code and experiment assets used for a
self-evolution training workflow built on ULTRA. Only two graph augmentation
rules are retained:

1. **Single-relation symmetry**: if relation `r` is sufficiently symmetric,
   `(h, r, t)` proposes `(t, r, h)`.
2. **Relation co-occurrence completion**: if `r1` and `r2` frequently occur on
   the same `(h, t)` pairs, either relation can propose the missing counterpart.

Every structural candidate is scored by the current ULTRA model before it is
added to the next iteration's training graph. Inverse-relation-pair discovery,
triangle closure, and mode-switching code have been removed.

## Repository layout

```text
.
|-- script/self_evolution.py       # augmentation, training, and evaluation
|-- ultra/                         # minimal ULTRA model/runtime modules
|-- config/                        # four NELL experiment configurations
|-- kg-datasets/                   # train/valid/test triples used by configs
|-- ckpts/ultra_50g.pth            # pretrained initialization
|-- run.sh                         # portable four-dataset launcher
`-- requirements.txt
```

## Data format

Each dataset directory must contain `train.txt`, `valid.txt`, and `test.txt`.
Every non-empty line has three whitespace-separated fields:

```text
head_entity relation tail_entity
```

The loader is path-independent: adding `kg-datasets/my_graph/` and setting
`dataset.graphs: [my_graph]` in a config is sufficient.

## Installation

The original environment used Python 3.9, PyTorch 2.1, PyTorch Geometric 2.4,
and CUDA 11.8. Install the CUDA-compatible PyTorch, PyG, and torch-scatter
wheels for your machine first, then install the remaining requirements:

```bash
pip install -r requirements.txt
```

The custom RSPMM operator is compiled on first use and requires `ninja`. GPU
compilation additionally requires a compatible CUDA toolkit and `nvcc`.

## Running

Run all four included configurations on logical GPU 0:

```bash
bash run.sh
```

Run a single configuration:

```bash
python -u script/self_evolution.py \
  -c config/pretrain_nl_v1_ind.yaml \
  --gpus '[0]' \
  --ckpt ckpts/ultra_50g.pth
```

For CPU execution, pass `--gpus null`. Useful launcher overrides include:

```bash
GPU=1 MAX_ITERATIONS=3 SYM_RATIO_THRESHOLD=0.6 bash run.sh
```

The main augmentation options are:

- `--sym_ratio_threshold`: structural symmetry ratio.
- `--sym_score_threshold`: model-score threshold for symmetry candidates.
- `--cooccur_ratio_threshold`: bidirectional co-occurrence ratio.
- `--cooccur_score_threshold`: model-score threshold for co-occurrence candidates.
- `--min_cooccur_support`: minimum number of shared `(h, t)` pairs.
- `--max_iterations`: total self-evolution training iterations.
- `--patience`: early-stopping patience.

Setting both score thresholds to `0` makes the model filter permissive because
sigmoid scores are non-negative.

## Output and mutation warning

Each run creates a timestamped directory below `output/`. The self-evolution
step also rewrites the selected dataset's `train.txt`; a `train_best.txt`
snapshot is created for rollback. Keep a clean copy of the original data when
running new experiments or use version control.

## Attribution and redistribution checklist

The model implementation and checkpoint originate from the MIT-licensed
[ULTRA](https://github.com/DeepGraphLearning/ULTRA) project. Keep `LICENSE` and
cite the ULTRA authors when publishing results.

The bundled NELL splits and pretrained checkpoint are third-party research
artifacts. Before publishing this archive, independently verify that their
upstream dataset/model terms permit redistribution, and add the precise source
and citation for the split-generation procedure used in your work.
