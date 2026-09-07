# BBO Capstone Technical Justification & Literature Foundations Report
**Imperial College London - Capstone Project**

---

## 1. Technical Justification & Prior Research
Our Black-Box Optimization (BBO) framework is technically justified by the fundamental problem constraints: the 8 target functions have unknown continuous response surfaces, lack analytical gradient expressions, and impose a strict evaluation budget of one sample per week.

To solve this, we implemented a probabilistic surrogate-driven Bayesian Optimization architecture. Rather than relying on unguided grid search or heuristic local search, Gaussian Process (GP) regression constructs a continuous posterior probability distribution (predictive mean and epistemic variance) over the unit hypercube. This posterior powers Expected Improvement (EI) and Upper Confidence Bound (UCB) acquisition functions, which quantify the expected gain of sampling unvisited candidate coordinates.

In our Week 6 evaluations, this approach enabled record-breaking performance leaps, including Function 5 surging to y = 4397.95, Function 7 rising to y = 2.2333, Function 6 reaching y = -0.2166, and Function 3 advancing to y = -0.0048.

---

## 2. Academic Literature and Key Papers
Our software design is directly grounded in three seminal papers from Bayesian Optimization literature:

1. Mockus, Tiesis, and Zilinskas (1978) - "On Bayesian methods for seeking the extremum":
This foundational work introduced the Expected Improvement (EI) acquisition function. We incorporated Mockus's formulation with a range-scaled jitter parameter to balance local exploitation around known peaks with global exploration in high-variance regions.

2. Snoek, Larochelle, and Adams (2012) - "Practical Bayesian Optimization of Machine Learning Algorithms" (NIPS):
Snoek et al. established that Matérn 5/2 covariance kernels provide realistic smoothness assumptions for physical black-box functions, outperforming rigid squared exponential (RBF) kernels. Following their methodology, we implemented Matérn 5/2 and Matérn 3/2 kernels with Maximum Marginal Likelihood optimization, enabling robust lengthscale adaptation across varying dimensions.

3. Shahriari et al. (2016) - "Taking the Human Out of the Loop: A Review of Bayesian Optimization" (IEEE):
Shahriari et al. emphasized that no single surrogate model dominates across all objective topologies. Inspired by this review, we built a 5-fold cross-validation benchmarking suite that continuously ranks Gaussian Processes, Multi-Layer Perceptron Neural Networks, Extra Trees, Random Forests, and Polynomial Ridge Regression.

---

## 3. Libraries, Frameworks, and Architectural Trade-Offs
Our implementation is built on scikit-learn, scipy, numpy, pandas, and matplotlib:

- scikit-learn: Provides robust, deterministic implementations of GaussianProcessRegressor (with L-BFGS-B kernel optimization), MLPRegressor, ExtraTreesRegressor, and Polynomial Ridge pipelines.
- scipy: Supplies exact normal cumulative distribution and probability density calculations required for closed-form Expected Improvement.

Trade-Off Analysis:
We evaluated heavy deep learning frameworks like PyTorch and TensorFlow but selected scikit-learn for our core surrogate engine. In a small-sample regime (15 to 45 observations per function), exact Gaussian Process inference executes in milliseconds on standard CPU hardware without GPU overhead or stochastic initialization instability. Furthermore, standard neural network frameworks do not natively output calibrated posterior variance, which is strictly required to compute Expected Improvement.

---

## 4. Documentation and GitHub Presentation Strategy
To ensure transparent communication for peers, facilitators, and employers, we organized our GitHub repository around clear software engineering principles:

- Executive README: Features an architectural flowchart, environment quickstart instructions, and a multi-week milestone progression table highlighting historical record gains across all functions.
- Telemetry & Visualization Artifacts: Every weekly run automatically exports 4-panel diagnostic plots (contour maps, 1D slice profiles, and feature importances) to a dedicated visualizations directory and logs cross-validation metrics to structured JSON files.
- Methodological Transparency: Dedicated markdown reports detail our technical justifications, exploration policies, and hyperparameter choices.

---

## 5. Future Research, Benchmarks, and Software Refinements
To further refine our strategy in higher-dimensional spaces (Function 7 in 6D and Function 8 in 8D), we plan to consult the following advanced sources:

- TuRBO (Trust Region Bayesian Optimization by Eriksson et al., 2019): Implements local Gaussian Process surrogates bounded inside dynamic trust regions to prevent sample dispersion in high dimensions.
- High-Dimensional BBO Literature (REMBO by Wang et al., 2016): Utilizes random linear embeddings to project high-dimensional search spaces onto lower-dimensional active subspaces.
- Modern BBO Frameworks: Exploring BoTorch (PyTorch-backed Bayesian optimization) and Optuna for parallelized Monte Carlo acquisition and automated surrogate hyperparameter tuning.
