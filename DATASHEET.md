# Datasheet: Black-Box Optimisation Capstone Dataset

Following the documentation framework established by Gebru et al. (2021) and adapted for sequential experimental data in Mini-lesson 21.1, this datasheet details the provenance, composition, collection, and governance of the black-box optimisation dataset.

---

## 1. Motivation

- **Purpose**: This dataset was constructed to benchmark, evaluate, and refine sample-efficient black-box optimisation algorithms under severe query constraints where functional forms, gradients, and noise distributions are entirely unobservable.
- **Supported Task**: Sequential Model-Based Global Optimisation (SMBO) and active learning across eight synthetic and empirical response surfaces spanning dimensions two through eight.
- **Creator & Maintainer**: Created and curated by the capstone engineering team at Imperial College London / Emeritus Programme.

---

## 2. Composition

- **Dataset Content**: The dataset consists of eight distinct function subsets, each representing a continuous response surface bounded within a unit hypercube:
  - Function 1 (2D): 19 evaluations (Smooth, plateau-like landscape with global maximum near 0.0).
  - Function 2 (2D): 19 evaluations (Bimodal surface with global peak at y = 0.676790).
  - Function 3 (3D): 24 evaluations (Rough surface with central peak mode at y = -0.004807).
  - Function 4 (4D): 39 evaluations (Needle-in-a-haystack positive island surrounded by deep negative valleys; best y = 0.367529).
  - Function 5 (4D): 29 evaluations (Exponential boundary ridge ascending sharply towards domain vertices; best y = 8662.482500).
  - Function 6 (5D): 29 evaluations (Non-linear plateau; best y = -0.208432).
  - Function 7 (6D): 39 evaluations (Multi-modal oscillatory surface; best y = 2.233318).
  - Function 8 (8D): 49 evaluations (High-dimensional manifold with low-dimensional active subspaces; all-time peak at y = 9.956667).
- **Total Volume**: 247 coordinate-evaluation pairs across 8 functions.
- **Data Format**: Serialised binary NumPy arrays (`initial_inputs.npy` of shape `(N, d)` and `initial_outputs.npy` of shape `(N,)`) alongside JSON telemetry logs (`weekly_progress_history.json`).
- **Gaps & Sparsity**: Due to strict physical evaluation budgets (one query per function per week across 10 rounds), the search spaces—especially in dimensions 6 through 8—are sparsely sampled, with evaluations deliberately concentrated along high-performing ridges and domain boundaries.

---

## 3. Collection Process

- **Sampling Timeline**: The dataset was acquired over ten successive weekly iterations, beginning from pseudo-random Latin Hypercube initialisations (weeks 1 to 3) and transitioning into adaptive Bayesian acquisition (weeks 4 to 10).
- **Query Generation Strategy**:
  - Initial Phase: Quasi-random space-filling Latin Hypercube Sampling (QMC) to establish baseline variance.
  - Active Optimisation Phase: Acquisition maximisation using Expected Improvement (EI) and Upper Confidence Bound (UCB) derived from surrogate Gaussian Processes equipped with Automatic Relevance Determination (ARD).
  - Candidate Pooling: 70% global Latin Hypercube candidates combined with 30% local Gaussian perturbations around historical optima, filtered with a minimum Euclidean distance threshold (min_dist = 0.005) to eliminate duplicate evaluations.

---

## 4. Preprocessing and Intended Uses

- **Transformations**: Raw inputs are strictly constrained to the continuous domain (0.000000 to 1.000000). Gaussian Process surrogates internally employ zero-mean and unit-variance target standardisation (`normalize_y=True`), whilst tree-based regressors operate directly on raw response values.
- **Intended Use Cases**:
  - Benchmarking surrogate model families (Gaussian Processes, ensembles, neural networks) under extreme data scarcity.
  - Studying active subspace identification in moderately high-dimensional black-box landscapes.
  - Validating acquisition functions under severe risk-reward trade-offs.
- **Inappropriate Use Cases**:
  - Training deep parametric neural networks without regularisation, as sample sizes are insufficient to prevent catastrophic over-fitting.
  - Assuming global stationarity or unbounded coordinate extrapolation beyond the unit hypercube.

---

## 5. Distribution and Maintenance

- **Hosting & Repository**: Maintained within the project Git repository alongside execution scripts (`bbo.py`) and diagnostic visualisations.
- **Licensing & Terms**: Educational and academic open-access research license; available for peer review and algorithmic reproduction.
- **Update Cadence**: Updated iteratively upon the completion of each weekly physical evaluation cycle.
