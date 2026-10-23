# Datasheet: Black-Box Optimisation Capstone Dataset

Following the documentation framework established by Gebru et al. (2021) and adapted for sequential experimental data in Module 21, this datasheet details the provenance, composition, collection process, preprocessing, and governance of our complete Black-Box Optimisation (BBO) capstone dataset through the final submission round (Week 13 / Module 24).

---

## 1. Motivation

- **Purpose**: This dataset was constructed to benchmark, evaluate, and refine sample-efficient black-box optimisation algorithms under strict physical query constraints where underlying functional forms, analytical gradients, and noise distributions are unobservable.
- **Supported Task**: Sequential Model-Based Global Optimisation (SMBO), active learning, and active subspace discovery across eight continuous response surfaces spanning two to eight input dimensions.
- **Creator & Maintainer**: Curated by the Capstone Engineering Team within the Imperial College London Machine Learning & AI Programme.

---

## 2. Composition

- **Dataset Content**: The dataset comprises eight distinct function subsets (`function_1` through `function_8`), each mapping a continuous coordinate vector within the unit hypercube `(0.0, 1.0)^d` to a scalar response `y`:
  - **Function 1 (2D)**: 22 evaluations — Flat zero-response plain with an extremely narrow needle-like anomaly (`best y = 0.000000`, baseline `7.71e-16`).
  - **Function 2 (2D)**: 22 evaluations — Noisy multi-modal radiation/sensor response landscape with a verified high plateau (`best y = 0.676790`).
  - **Function 3 (3D)**: 27 evaluations — Dose-response surface with an isolated central optimum near zero (`best y = -0.004807`).
  - **Function 4 (4D)**: 42 evaluations — Complex non-convex surface featuring a steep positive centroid island surrounded by deep negative cliffs (`best y = +0.367529`).
  - **Function 5 (4D)**: 32 evaluations — Steep exponential chemical yield surface ascending sharply to the upper domain vertex `(1.0, 1.0, 1.0, 1.0)` (`best y = 8662.482500`).
  - **Function 6 (5D)**: 32 evaluations — Five-dimensional formulation landscape with strong parameter sensitivity (`best y = -0.089540`, all-time record).
  - **Function 7 (6D)**: 42 evaluations — Six-dimensional hyperparameter surface with a pronounced high-performing basin (`best y = 3.227650`, all-time record).
  - **Function 8 (8D)**: 52 evaluations — Eight-dimensional manifold governed by a low-dimensional active subspace (`best y = 9.981970`, all-time record).
- **Total Volume**: **271 coordinate-evaluation pairs** across the 8 benchmark functions (expanded from 175 initial observations plus 96 sequential weekly queries across 12 evaluation cycles).
- **Data Size & GitHub Hosting Policy**: All binary data files (`initial_inputs.npy` and `initial_outputs.npy`) total less than 50 KB combined and are stored directly inside `function_1/` through `function_8/` for immediate reproducibility. No external large-file storage is required.
- **Data Format**: Serialised binary NumPy arrays (`initial_inputs.npy` of shape `(N, d)` and `initial_outputs.npy` of shape `(N,)`) paired with structured JSON telemetry (`weekly_progress_history.json` and `week1_summary.json` through `week13_summary.json`).
- **Sampling Sparsity & Selection Bias**: Because sequential queries were selected to maximise Expected Improvement (EI) and Upper Confidence Bound (UCB), the dataset is intentionally non-uniform (epistemically biased toward high-yield ridges and active boundaries). Users should not treat these samples as independent and identically distributed (i.i.d.) uniform draws.

---

## 3. Collection Process

- **Sampling Timeline**: Collected sequentially across twelve weekly evaluation rounds (Weeks 1 to 13 / Modules 12 to 24), adding strictly one evaluation per function per round via the course evaluation portal.
- **Query Generation Evolution**:
  - **Weeks 1–3 (Exploration Baseline)**: Space-filling quasi-random sampling and isotropic Gaussian Process / ensemble regression baselines.
  - **Weeks 4–6 (Multi-Model Cross-Validation)**: Automated 5-fold cross-validation tournament comparing Gaussian Processes, Neural Networks (MLP), Random Forests, Extra Trees, Gradient Boosting, and Polynomial Ridge Regression.
  - **Weeks 7–13 (Anisotropic ARD & Hybrid Trust-Region Exploitation)**: Gaussian Processes equipped with Automatic Relevance Determination (ARD) over a 30,000-point hybrid candidate pool (70% global Latin Hypercube Sampling via `scipy.stats.qmc` and 30% multi-scale local Gaussian perturbations around the top-3 historical observations), filtered by a minimum Euclidean distance threshold (`min_dist = 0.005`) to prevent redundant queries.

---

## 4. Preprocessing, Cleaning, and Intended Uses

- **Sanitisation & Bounds**: Prior to surrogate fitting, all input arrays `X` and target vectors `y` are validated for finite numerical values (`NaN`/`Inf` inspection) and strictly clamped to the continuous unit hypercube `(0.000000, 1.000000)^d`.
- **Target Scaling**: Gaussian Process surrogates apply internal zero-mean, unit-variance target standardisation (`normalize_y=True`), whilst neural network and polynomial ridge pipelines apply `StandardScaler` transformations across features.
- **Intended Use Cases**:
  - Benchmarking sample-efficient Bayesian optimisation and active learning algorithms under tight evaluation budgets.
  - Studying dimensionality reduction, Principal Component Analysis (PCA), and ARD lengthscale pruning in 2D to 8D spaces.
  - Educational reproduction of sequential decision-making and exploration-exploitation trade-offs.
- **Inappropriate Use Cases**:
  - Training unregularised, high-capacity deep neural networks expecting uniform global coverage.
  - Extrapolating predictions outside the bounded unit hypercube `(0.0, 1.0)^d`.

---

## 5. Distribution and Maintenance

- **Repository Location**: Hosted in the root repository under `function_1/` through `function_8/`, with full historical metrics tracked in `weekly_progress_history.json` and `bbo_weekly_tracking.md`.
- **Licensing & Access**: Released for academic, educational, and portfolio review under standard open-source reproducibility guidelines.
- **Maintenance Status**: Finalised upon completion of Week 13 (Module 24/25) capstone closure.
