# Model Card: Adaptive Multi-Surrogate Bayesian Optimisation with ARD

Following the standardised model card framework proposed by Mitchell et al. (2019) and introduced in Module 21, this document details the architecture, intended use, multi-week evolution, final empirical performance (through Week 13 / Module 24), limitations, and governance of our Black-Box Optimisation system.

---

## 1. Model Details

- **Model Name**: Adaptive Multi-Surrogate Bayesian Optimiser with Automatic Relevance Determination (`BBO-ARD-MSB`).
- **Model Type**: Sequential Model-Based Global Optimisation (SMBO) and Active Learning Pipeline.
- **Model Version**: 3.0 — Final Capstone Release (implemented in [`bbo.py`](bbo.py) and interactive notebook [`BBO_Capstone_Walkthrough.ipynb`](BBO_Capstone_Walkthrough.ipynb)).
- **Developers**: Capstone Engineering Team, Imperial College London Machine Learning & AI Programme.
- **Core Algorithmic Components**:
  - **Primary Probabilistic Surrogates**: Anisotropic Gaussian Process Regressors (`GaussianProcessRegressor`) using dimension-specific Automatic Relevance Determination (ARD) kernels (`Matérn nu=1.5`, `Matérn nu=2.5`, and `RBF`) scaled by a constant kernel and optimised via 15 restarts of `L-BFGS-B`.
  - **Cross-Validation Benchmark Suite**: 5-fold cross-validation tournament evaluating ARD Gaussian Processes alongside `ExtraTreesRegressor` (100 trees), `RandomForestRegressor` (100 trees), `GradientBoostingRegressor` (50 trees), quadratic `PolynomialFeatures + Ridge`, and a two-layer `MLPRegressor` `(64, 32)` with ReLU activation and L2 regularisation (`alpha = 1e-3`).
  - **Candidate Generation & Acquisition**: Hybrid 30,000-point candidate generator (70% Quasi-Monte Carlo Latin Hypercube Sampling + 30% multi-scale Gaussian trust-region perturbations around the top-3 historical maxima), Euclidean distance de-duplication (`min_dist = 0.005`), and Expected Improvement (`xi = 0.01 * y_range`) with automatic Upper Confidence Bound (`beta = 2.576`) fallback.

---

## 2. Intended Use

- **Primary Application**: Derivative-free global optimisation of expensive, non-convex, black-box functions over bounded continuous domains `(0.0, 1.0)^d` where query budgets are severely restricted (10 to 55 total evaluations).
- **Target Audience**: Machine learning engineers, experimental scientists, and peer reviewers exploring sample-efficient optimisation, hyperparameter tuning, and automated experimental design.
- **Out-of-Scope Uses**:
  - High-dimensional raw perceptual inputs (e.g. raw pixel arrays or text tokens) without prior latent embedding.
  - Unconstrained real-time industrial control without hardware safety interlocks, as exploratory queries can intentionally probe high-variance boundary regions.

---

## 3. Methodological Evolution (Weeks 1 to 13)

1. **Phase 1 — Baseline Exploration & Conditioning (Weeks 1–3)**: Deployed isotropic Gaussian Processes and tree ensembles over random grids. Identified covariance conditioning bottlenecks and introduced target standardisation (`normalize_y=True`) and adaptive Cholesky jitter (`alpha`).
2. **Phase 2 — Multi-Surrogate Cross-Validation & Neural Baselines (Weeks 4–6)**: Built an automated 5-fold CV benchmarking harness comparing GP kernels against regularized MLPs, Polynomial Ridge, and tree ensembles, binding the lowest-MSE GP kernel per function to preserve calibrated posterior variance.
3. **Phase 3 — Anisotropic ARD & Hybrid Trust-Region Sampling (Weeks 7–10)**: Replaced scalar lengthscales with vector ARD lengthscales `(l_1, ..., l_d)` to prune inactive dimensions in 5D–8D spaces, paired with a 70/30 QMC Latin Hypercube and local Gaussian trust-region sampler. Unlocked the `8662.482500` vertex peak on Function 5.
4. **Phase 4 — Unsupervised Subspace Diagnostics & Horizon-Aware Exploitation (Weeks 11–13)**: Integrated clustering, PCA active-subspace inspection, and RL-inspired value-refinement strategies, achieving three consecutive all-time record breakthroughs on Functions 6, 7, and 8 in the final rounds.

---

## 4. Final Empirical Performance (Week 13 / 271 Total Evaluations)

Evaluated via 5-fold cross-validation (`CV_MSE_Mean` and `CV_R2_Mean`) and cumulative global optimization gain from Week 1 to Week 13:

| Function | Dim | Samples (Init -> Final) | Winning GP Kernel | Final Best `y` | Total Gain | Final Topographical & Performance Summary |
| :--- | :---: | :---: | :--- | :---: | :---: | :--- |
| **Function 1** | 2D | 10 -> 22 | GP (Matérn 2.5 ARD) | `0.000000` | Baseline | Mapped upper boundary; flat zero plateau confirmed (`CV MSE < 1e-6`). |
| **Function 2** | 2D | 10 -> 22 | GP (Matérn 1.5 ARD) | **`0.676790`** | `+0.0656` | High plateau isolated and verified (`CV R2 = 0.252` under observational noise). |
| **Function 3** | 3D | 15 -> 27 | GP (Matérn 1.5 ARD) | **`-0.004807`** | `+0.0300` | Central peak mode tightly bounded (`CV MSE = 0.0061`). |
| **Function 4** | 4D | 30 -> 42 | GP (Matérn 1.5 ARD) | **`+0.367529`** | `+4.3931` | Isolated narrow positive island (`CV R2 = 0.946`); verified sharp cliff boundaries. |
| **Function 5** | 4D | 20 -> 32 | GP (Matérn 1.5 ARD) | **`8662.482500`** | **`+7573.62`** | Colossal global peak reached at upper vertex `(1.0, 1.0, 1.0, 1.0)`. |
| **Function 6** | 5D | 20 -> 32 | GP (Matérn 1.5 ARD) | **`-0.089540`** | `+0.6247` | All-time record high in Week 13 (`CV R2 = 0.788`). |
| **Function 7** | 6D | 30 -> 42 | GP (RBF ARD) | **`3.227650`** | `+1.8627` | All-time record high in Week 13 (`CV R2 = 0.678`). |
| **Function 8** | 8D | 40 -> 52 | GP (Matérn 2.5 ARD) | **`9.981970`** | `+0.3835` | All-time record peak in Week 13 (`CV R2 = 0.989` via ARD active subspace). |

---

## 5. Limitations, Biases, and Trade-Offs

- **Needle-in-a-Haystack Blindness (Function 1)**: When a function is identically zero across 99% of its domain and exhibits a discontinuity narrower than the sampling grid, Gaussian Process lengthscales flatten out and Expected Improvement collapses (`< 1e-12`), requiring UCB fallback or space-filling exploration.
- **Sharp Cliff Sensitivity (Function 4)**: Stepping even `0.05` units outside a narrow positive centroid in 4D can drop the objective from `+0.367` to `-2.360`, highlighting the risk of late-stage exploratory variance seeking.
- **Active Exploitation Bias**: Because 30% of candidates cluster around the top-3 historical points, the surrogate achieves high fidelity near known optima but retains higher epistemic uncertainty in unvisited corners of 6D–8D space.

---

## 6. Ethical Considerations, Transparency & Reproducibility

- **Deterministic Reproducibility**: All stochastic components (K-Fold splits, L-BFGS-B restarts, Latin Hypercube samplers, and local Gaussian perturbations) are explicitly seeded (`random_state=42` / `seed=42`).
- **Safety in Physical Deployment**: When adapting this pipeline to wet-lab chemistry, clinical dosing, or industrial hardware tuning, practitioners must pair acquisition maximisation with explicit safety constraints (e.g. SafeOpt / constrained Bayesian optimisation) so that exploratory probes never violate safe operating envelopes.
