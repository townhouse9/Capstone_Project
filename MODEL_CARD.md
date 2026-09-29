# Model Card: Adaptive Multi-Surrogate Bayesian Optimisation with ARD

Following the standardized model card framework proposed by Mitchell et al. (2019) and structured in Mini-lesson 21.2, this document describes the architecture, intended application, multi-round evolution, empirical performance, and governance of the black-box optimisation system.

---

## 1. Model Details

- **Model Name**: Adaptive Multi-Surrogate Bayesian Optimiser with Automatic Relevance Determination (BBO-ARD-MSB).
- **Model Type**: Sequential Model-Based Global Optimisation (SMBO) / Active Learning Pipeline.
- **Model Version**: 2.1 (Modular production release in `bbo.py`).
- **Developers**: Capstone Engineering Team, Imperial College London / Emeritus Programme.
- **Underlying Algorithms**:
  - Primary Probabilistic Surrogate: Gaussian Process Regression with dimension-specific anisotropic Matérn (nu = 1.5, 2.5) and Radial Basis Function (RBF) kernels fitted via multi-restart L-BFGS-B maximum marginal likelihood estimation.
  - Benchmarking Regressors: Random Forest, Extra Trees, Gradient Boosting, Polynomial Ridge (degree = 2), and Multi-Layer Perceptron (MLP) Neural Networks evaluated through 5-fold cross-validation.
  - Acquisition Engines: Expected Improvement (EI) with range-scaled exploration parameter (xi = 0.01 * y_range), transitioning to Upper Confidence Bound (UCB, beta = 2.576) in flat regions.

---

## 2. Intended Use

- **Primary Application**: Global optimisation of continuous, non-convex, derivative-free black-box functions evaluated under constrained query budgets where individual function evaluations are expensive, noisy, or time-delayed.
- **Suitable Domains**: Hyperparameter optimisation in machine learning, engineering design search, algorithmic tuning, and experimental design in physical laboratories.
- **Out-of-Scope / Prohibited Uses**:
  - High-throughput discrete combinatorial optimisation without continuous relaxation.
  - Safety-critical real-time control systems where millisecond inference is mandatory and sub-optimal queries induce catastrophic physical failure.
  - Unbounded continuous optimisation without domain constraints.

---

## 3. Methodological Details and Multi-Round Evolution

Across ten iterative rounds, the optimisation approach evolved across three major phases:
1. **Initial Space-Filling & Exploration (Rounds 1–3)**: Established initial landscape coverage using Latin Hypercube Sampling (QMC), benchmarking random forests against isotropic Gaussian Processes.
2. **Bayesian Transition & Surrogate Benchmarking (Rounds 4–6)**: Formalised automated 5-fold cross-validation across competing regressors to determine the best surrogate family per function. Introduced Expected Improvement and explored neural network surrogate approximations.
3. **Anisotropic ARD, Trust-Region Perturbation, and Active Subspace Exploitation (Rounds 7–10)**: Upgraded Gaussian Processes to Automatic Relevance Determination (ARD) to learn dimension-specific lengthscales, implemented TuRBO-inspired candidate generation (70% global LHS, 30% local Gaussian perturbation around top-3 historical optima), and enforced a minimum Euclidean distance filter (min_dist = 0.005) to eliminate redundant sampling.

---

## 4. Empirical Performance Across Benchmark Functions

Surrogate model selection was guided by 5-fold cross-validation Mean Squared Error (CV-MSE) and coefficient of determination (CV-R2). Performance culminated in four simultaneous global records in Round 10:

- **Function 1 (2D)**: Mapped plateau boundary; current best y = 0.000000.
- **Function 2 (2D)**: Selected Matérn 1.5 ARD (CV-R2 = 0.423); achieved new global best y = 0.676790.
- **Function 3 (3D)**: Selected Matérn 1.5 ARD (CV-MSE = 0.0069); bounded central peak mode at y = -0.004807.
- **Function 4 (4D)**: Selected Matérn 1.5 ARD (CV-R2 = 0.949); isolated needle-in-a-haystack positive peak island at y = 0.367529, successfully recovering to y = 0.013954 after deep valley descent.
- **Function 5 (4D)**: Selected Matérn 1.5 ARD; surged to an all-time record of y = 8662.482500 by exploiting the upper vertex (1.0, 1.0, 1.0, 1.0).
- **Function 6 (5D)**: Selected Matérn 1.5 ARD (CV-R2 = 0.871); reached new global best y = -0.208432.
- **Function 7 (6D)**: Selected Matérn 1.5 ARD (CV-R2 = 0.558); sustained high-yield basin at y = 2.233318.
- **Function 8 (8D)**: Selected Matérn 2.5 ARD (CV-R2 = 0.987); unlocked an all-time record of y = 9.956667 by exploiting low-dimensional active subspaces.

---

## 5. Assumptions, Constraints, and Limitations

- **Stationarity & Continuity**: The surrogate assumes objective functions exhibit spatial autocorrelation and continuous local gradients. Discontinuous step changes or Dirac-like delta functions violate these assumptions.
- **Curse of Dimensionality & Sparsity**: With 49 samples in 8 dimensions, Gaussian Processes risk overfitting lengthscales to early points, leaving vast volumes of the hypercube unvisited.
- **Exploitation Bias**: The acquisition engine heavily favours refining confirmed high-yield basins, creating epistemic blind spots in unexplored interior regions.

---

## 6. Ethical Considerations & Reproducibility

- **Transparency and Open Science**: To ensure complete scientific reproducibility, all random seeds (seed 42), candidate generation pipelines, hyperparameter bounds, and raw telemetry are preserved in the open repository without proprietary obfuscation.
- **Responsible Deployment**: Users adapting this system for real-world automated experimentation (e.g. chemical synthesis or robotics) must integrate hardware safety limits and domain-specific constraint checks to prevent physical harm from unconstrained acquisition exploration.

---

## 7. Architectural Reflection

The decision-making architecture dynamically combines multi-model empirical cross-validation with probabilistic active learning, ensuring every query is statistically justified rather than heuristically guessed. While incorporating deeper theoretical descriptions (such as full Hessian matrix derivations) could be considered, the current structure provides an optimal balance of operational clarity, mathematical rigor, and reproducible implementation for researchers and industry practitioners alike.
