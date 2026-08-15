# SARA: Structure-Aware Relational Augmentation for Knowledge Graph Completion

> This repository currently provides a preliminary version of the artifact, and we will continue to expand and improve the code, documentation, and reproducibility resources in subsequent updates.

This repository contains the implementation and experimental artifacts for **SARA**, a lightweight and model-agnostic knowledge graph enrichment framework for knowledge graph completion (KGC).

SARA augments an observed knowledge graph using relation-level structural regularities before the downstream KGC model is trained or evaluated. It does **not** modify the backbone architecture, training objective, or scoring function.

## Overview

SARA contains two complementary augmentation strategies:

1. **Symmetric Relational Augmentation**
   For a relation that exhibits sufficiently strong symmetric behavior in the observed graph, SARA adds missing reversed triples.

2. **Co-occurrence Relational Augmentation**
   For two relations that frequently co-occur on the same ordered entity pairs, SARA adds the missing co-occurring relation when the corresponding relation-level selection criteria are satisfied.

The resulting augmented graph can be used as a drop-in preprocessing step for different KGC backbones.

## Key Properties

* **Model-agnostic:** no modification to the downstream KGC architecture or scoring function.
* **Non-parametric:** SARA itself introduces no learnable parameters.
* **Non-iterative:** structural statistics are computed directly from the observed graph.
* **Structure-aware:** augmented triples are generated from relation-level symmetry and co-occurrence patterns.
* **Leakage-aware:** augmentation uses only the graph observable under the corresponding evaluation protocol; held-out validation/test triples are never used to estimate augmentation statistics or generate augmented triples.

## Experimental Scope

The main experiments evaluate SARA on **27 inductive and fully inductive KGC benchmarks** with the following backbones:

* ULTRA
* TRIX
* MOTIF
* Flock

The evaluation uses standard KGC metrics including:

* MRR
* Hits@1
* Hits@3
* Hits@10

The manuscript experiments were implemented in PyTorch and conducted on two NVIDIA H800 GPUs. The original KGFM backbones use their corresponding pretrained checkpoints and default backbone hyperparameters.

### Additional Experiments Added During Revision

To evaluate SARA beyond KGFM-style backbones, we additionally include:

* **TransE** — conventional embedding-based KGC
* **GraIL** — inductive subgraph reasoning

We also include stronger KG augmentation comparisons with:

* **NNMFAug**
* **KnowAug**

The additional experiments are conducted on NELL-v1, NELL-v2, NELL-v3, and NELL-v4.

For **TransE**, we use a **target-graph transductive protocol**: TransE is fitted on the observed message graph of each target test graph and evaluated on held-out test triples from the same graph. SARA is applied only to the observed message graph. Held-out test triples are not used for augmentation or model fitting.

For **GraIL**, we follow the standard inductive evaluation protocol.

## Main Results

Average performance reported for the four main backbones:

| Backbone | Vanilla MRR | SARA MRR |
| -------- | ----------: | -------: |
| ULTRA    |        46.9 |     49.2 |
| MOTIF    |        46.3 |     48.5 |
| TRIX     |        41.2 |     46.5 |
| Flock    |        47.8 |     49.3 |

The same-size Random baseline adds exactly the same number of triples as SARA for each dataset, but generally degrades performance. This controls for augmentation volume and shows that simply adding more triples is insufficient.

Across all datasets, the training-set-size-weighted augmentation ratio of SARA is approximately **13.1%**.

### Additional Backbone Results on NELL-v1–v4

| Backbone | Vanilla MRR | SARA MRR | Vanilla Hits@1 | SARA Hits@1 |
| -------- | ----------: | -------: | -------------: | ----------: |
| TransE   |        41.0 |     53.9 |           25.8 |        34.4 |
| GraIL    |        72.8 |     82.1 |           64.5 |        75.8 |

### Additional Augmentation Baselines on NELL-v1–v4 with ULTRA

| Method    |  MRR | Hits@1 |
| --------- | ---: | -----: |
| ULTRA     | 66.2 |   56.6 |
| + NNMFAug | 65.8 |   56.0 |
| + KnowAug | 68.5 |   60.4 |
| + SARA    | 73.3 |   68.0 |

## Repository Structure

The exact paths below should be adjusted to match the final repository layout.

```text
.
├── README.md
├── requirements.txt
├── configs/
│   ├── sara/
│   └── backbones/
├── data/
│   └── ...
├── src/
│   ├── augmentation/
│   │   ├── symmetry.py
│   │   └── cooccurrence.py
│   ├── preprocess.py
│   └── utils.py
├── scripts/
│   ├── run_sara.sh
│   ├── run_ultra.sh
│   ├── run_trix.sh
│   ├── run_motif.sh
│   ├── run_flock.sh
│   ├── run_transe.sh
│   └── run_grail.sh
└── results/
    └── ...
```

If your repository uses different filenames, replace this tree and the example commands below with the actual paths before release.

## Environment

We recommend creating an isolated Python environment.

```bash
conda create -n sara python=<PYTHON_VERSION> -y
conda activate sara

pip install -r requirements.txt
```

For exact reproducibility, the released artifact should record at least:

* Python version
* PyTorch version
* CUDA version
* GPU model
* versions/commits of external backbone repositories
* pretrained checkpoint identifiers

## Data Preparation

SARA operates on triples of the form

```text
(head, relation, tail)
```

For each dataset, keep the observable graph separated from held-out validation/test triples.

A recommended directory organization is:

```text
data/<dataset>/
├── train.txt
├── valid.txt
├── test.txt
└── message.txt
```

The exact filenames may differ across backbone implementations.

### No-Data-Leakage Requirement

SARA statistics must be computed only from the graph available under the corresponding protocol.

For training-graph augmentation:

```text
T_obs = T_train
```

For a target/support/message-graph experiment:

```text
T_obs = T_msg
```

Held-out validation/test triples must not be used to:

* compute symmetry confidence;
* compute relation co-occurrence confidence;
* select relation patterns;
* generate augmented triples;
* fit TransE in the target-graph transductive experiment.

## SARA Augmentation

Let (T_{\mathrm{obs}}) denote the observable graph.

### Symmetric Relational Augmentation

For relation (r), define

[
N_r = {(h,t)\mid(h,r,t)\in T_{\mathrm{obs}}}.
]

The empirical symmetry confidence is

[
C_{\mathrm{sym}}(r)
===================

\frac{
|{(h,t)\in N_r:(t,r,h)\in T_{\mathrm{obs}}}|
}{
|N_r|
}.
]

If

[
C_{\mathrm{sym}}(r)\ge \delta_{\mathrm{sym}},
]

missing reversed triples are added.

### Co-occurrence Relational Augmentation

For relations (r_i) and (r_j),

[
C_{\mathrm{co}}(r_i\rightarrow r_j)
===================================

\frac{|N_{r_i}\cap N_{r_j}|}{|N_{r_i}|}.
]

SARA accepts a relation pair only when the co-occurrence selection criterion is satisfied in both directions, then adds missing co-occurring relation triples.

## Running SARA

The final artifact should expose the augmentation step through a command similar to:

```bash
python <SARA_ENTRYPOINT> \
    --input <OBSERVED_GRAPH> \
    --output <AUGMENTED_GRAPH> \
    --delta_sym <DELTA_SYM> \
    --delta_co <DELTA_CO>
```

The output should contain

```text
T_aug = T_obs ∪ Δ_sym ∪ Δ_co
```

with duplicates removed.

For reproducibility, we recommend logging:

```text
dataset
|T_obs|
|Δ_sym|
|Δ_co|
|T_aug|
δ_sym
δ_co
random_seed
```

## Training and Evaluation

After augmentation, use the resulting graph exactly as the original backbone would use its observable graph.

Conceptually:

```bash
# 1. Generate augmented graph
python <SARA_ENTRYPOINT> ...

# 2. Train or fit the selected backbone
python <BACKBONE_ENTRYPOINT> \
    --data <AUGMENTED_GRAPH> \
    ...

# 3. Evaluate on the unchanged held-out split
python <EVAL_ENTRYPOINT> \
    ...
```

SARA does not require modification of the backbone architecture or scoring function.

## Reproducing the Main Comparisons

### Main KGFM Backbones

Evaluate

```text
Vanilla
+ Random
+ Reciprocal
+ SARA
```

for:

```text
ULTRA
TRIX
MOTIF
Flock
```

The Random baseline must use the **same number of augmented triples as SARA** on each dataset.

### Additional Augmentation Baselines

On NELL-v1–v4 with ULTRA:

```text
ULTRA
ULTRA + NNMFAug
ULTRA + KnowAug
ULTRA + SARA
```

### Additional Backbone Families

On NELL-v1–v4:

```text
TransE
TransE + SARA

GraIL
GraIL + SARA
```

## Threshold Sensitivity

The manuscript studies the effect of the relation-selection thresholds (\delta_{\mathrm{sym}}) and (\delta_{\mathrm{co}}).

A stricter high-confidence setting is additionally evaluated with

```text
δ_sym = 0.8
δ_co  = 0.8
```

When reproducing threshold experiments, report both:

1. downstream KGC performance; and
2. augmentation coverage,

for example,

[
\frac{|\Delta_{\mathrm{sym}}\cup\Delta_{\mathrm{co}}|}
{|T_{\mathrm{obs}}|}.
]

This makes the confidence–coverage trade-off explicit.

## Reproducibility Checklist

Before artifact release, please ensure that the repository contains:

* [ ] complete SARA source code;
* [ ] exact environment/dependency specification;
* [ ] dataset preparation scripts or download instructions;
* [ ] configs for all reported SARA experiments;
* [ ] configs for Random and Reciprocal baselines;
* [ ] configs for NNMFAug and KnowAug comparisons;
* [ ] configs for TransE and GraIL experiments;
* [ ] pretrained checkpoint instructions;
* [ ] fixed random seeds;
* [ ] commands for reproducing tables/figures;
* [ ] scripts for computing MRR and Hits@K;
* [ ] augmentation statistics (`|Δ_sym|`, `|Δ_co|`, augmentation ratio);
* [ ] expected outputs or reference result files.

## Notes on External Backbones

SARA is designed as graph-level preprocessing. External KGC implementations should therefore retain their original architecture, objective, and scoring function.

For each external backbone, the final artifact should record:

```text
repository URL:
commit hash:
checkpoint:
configuration:
dataset preprocessing:
evaluation command:
```

This is particularly important for ULTRA, TRIX, MOTIF, Flock, GraIL, and TransE so that the artifact can be reproduced independently.

## Artifact Evaluation Notes

The artifact is intended to support verification of the following claims:

1. SARA can be applied without modifying the downstream KGC architecture.
2. Symmetric and co-occurrence augmentation can be reproduced from the observable graph alone.
3. Random augmentation with the same augmentation volume does not reproduce SARA's gains.
4. SARA remains effective with multiple KGFM backbones.
5. SARA is also compatible with conventional embedding and inductive subgraph-reasoning backbones.
6. The added triples are generated without using held-out validation/test facts.

## Citation

This repository accompanies an anonymous manuscript currently under review.

Citation information will be added after the review process.

## License

Please add the license used for the released code and any dataset-specific licensing notices before public release.
