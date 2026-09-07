# Black-Box Optimization (BBO) - Week 7 (Module 18) Progress & Strategy Report
**Imperial College Capstone Project**

---

## 1. Week 7 Input Queries & Output Evaluation Summary

Following the evaluation of our Week 7 submissions, our optimization framework achieved an **astronomical new all-time record global high on Function 5**:
- Function 5 (4D): Surged to a staggering new record peak of y = 5893.206247 (a massive +1495.25 gain in a single round over the previous 4397.95 peak).
- Function 7 (6D): Sustained a high peak output of y = 2.177521 (confirming our high-yield basin near the all-time peak of 2.233318).
- Function 8 (8D): Maintained strong high-ridge performance at y = 9.793209 (near our all-time peak of 9.929387).

### 1.1 Progression Table (Week 1 to Week 7)

| Function | Dim | Samples (N) | Previous Max y | W6 Evaluated y | Current Best y | Status / Gain | Week 7 Proposed Query Submission (x1-x2-...-xn) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| Func 1 | 2D | 16 | 7.71e-16 | -1.267e-94 | 7.71e-16 | Upper boundary mapped | 0.542714-0.969849 |
| Func 2 | 2D | 16 | 0.664342 | 0.314274 | 0.664342 | High plateau confirmed | 0.628141-0.185930 |
| Func 3 | 3D | 21 | -0.004807 | -0.037348 | -0.004807 | Central mode bounded | 0.998561-0.024817-0.835244 |
| Func 4 | 4D | 36 | 0.367529 | -0.585530 | 0.367529 | Valley edge bounded | 0.003953-0.938100-0.995633-0.051384 |
| Func 5 | 4D | 26 | 4397.953210 | 5893.206247 | 5893.206247 | Astronomical Peak (+1495) | 0.951430-0.994819-0.711723-0.981144 |
| Func 6 | 5D | 26 | -0.216645 | -0.580961 | -0.216645 | High plateau mapped | 0.377067-0.365606-0.556370-0.849446-0.330112 |
| Func 7 | 6D | 36 | 2.233318 | 2.177521 | 2.233318 | High basin sustained | 0.160714-0.402693-0.443212-0.406155-0.337573-0.767210 |
| Func 8 | 8D | 46 | 9.929387 | 9.793209 | 9.929387 | High ridge sustained | 0.143181-0.053741-0.077646-0.278296-0.878902-0.438223-0.234852-0.471378 |

---

## 2. Portal-Ready Reflection Questions and Answers (Module 18)

### Question 1
**Prompt**: Which hyperparameters did you choose to tune, and why did you prioritise them?

**Answer**:
We prioritized hyperparameters governing spatial covariance decay, model regularization, and acquisition exploration pressure:
1. Gaussian Process Lengthscale Bounds and Smoothness (nu = 1.5 vs 2.5):
Lengthscale bounds (1e-2 to 1e2) determine how quickly spatial covariance decays across coordinate distances. Tuning lengthscale bounds and kernel smoothness was prioritized because black-box function response surfaces vary from smooth isotropic basins (Function 8) to sharp non-stationary ridges (Function 5).
2. Observational Noise Regularization (Alpha = 1e-4):
We tuned the GP alpha noise penalty to prevent ill-conditioned matrix inversion during L-BFGS-B marginal likelihood optimization while maintaining numerical stability across multi-scale target values.
3. Neural Network Architecture and Regularization:
For our Neural Network (MLP) surrogate, we tuned hidden layer topology (64, 32), ReLU activations, and L2 weight decay (alpha = 1e-3). Prioritizing L2 regularization prevented overparameterized networks from overfitting to sparse training data.
4. Acquisition Jitter (xi_frac = 0.01 * y_range):
Scaling expected improvement jitter relative to dynamic output range prevented acquisition collapse as peak values expanded into thousands.

---

### Question 2
**Prompt**: How has hyperparameter tuning changed your query strategy compared to earlier rounds?

**Answer**:
Hyperparameter tuning transformed our query strategy from unguided, isotropic exploratory sampling into targeted, high-yield ridge exploitation and variance reduction:

1. Dynamic Lengthscale Adaptation: In early rounds, uncalibrated isotropic kernels over-smoothed local peaks, directing queries toward uninformative central coordinates. Tuning lengthscale bounds allowed Gaussian Processes to capture steep anisotropic slopes. On Function 5, this enabled our query engine to track the ascending upper corner, unlocking successive peaks from 1088.86 to 2062.99, 2304.29, 2921.37, 4397.95, and ultimately y = 5893.21 this round.
2. Range-Scaled Acquisition: Scaling acquisition jitter dynamically prevented acquisition saturation as function ranges exploded. On Function 8 (8D), tuning lengthscales and bounds focused queries along the narrow active subspace (Dimensions 5 and 6), sustaining peak outputs near y = 9.93.

---

### Question 3
**Prompt**: Which tuning method(s) did you apply (manual adjustment, grid search, random search, Bayesian optimisation, Hyperband), and what trade-offs did you notice?

**Answer**:
We implemented a hybrid tuning methodology combining 5-Fold Cross-Validation Grid Benchmarking with Maximum Marginal Likelihood (L-BFGS-B) kernel optimization:

1. 5-Fold CV Grid Benchmarking: Each round, we automatically evaluated 8 distinct surrogate model families (GP Matérn 2.5, GP Matérn 1.5, GP RBF, Neural Network MLP, Extra Trees, Random Forest, Gradient Boosting, Polynomial Ridge) using K-Fold cross-validation MSE and R-squared metrics.
2. Maximum Marginal Likelihood GP Tuning: For probabilistic models, kernel lengthscales and signal variances were fitted via 15 restarts of L-BFGS-B optimization.

Trade-Off Analysis:
Manual hyperparameter tuning was prone to cognitive bias and failed to adapt to expanding datasets. Full grid search across neural network architectures was computationally expensive and risked overfitting tiny datasets (16 to 46 points). Our automated 5-fold cross-validation engine provided the optimal balance: instantaneous CPU execution, deterministic model selection, and zero overfitting risk.

---

### Question 4
**Prompt**: As your data set grows to 16 points, what limitations of your model become clearer through tuning (e.g. overfitting, irrelevant features, diminishing returns)?

**Answer**:
As our evaluation datasets expanded from 10 to 16 points in 2D (Functions 1 and 2) and up to 46 points in 8D (Function 8), tuning revealed three critical structural limitations:

1. The Curse of Dimensionality and Sparsity:
In 2D (16 points), observations begin mapping local surface curvature effectively. In 8D (Function 8, 46 points), sampling remains extremely sparse. Gaussian Process posterior uncertainty remains high across unvisited hypercube volume, making hyperparameter lengthscales sensitive to single outlier evaluations.
2. Model Capacity vs Small-Sample Overfitting:
Multi-Layer Perceptron (MLP) Neural Networks overfit severely on small 2D and 3D datasets, producing negative cross-validation R-squared scores due to overparameterization. However, on Function 6 (5D), MLP achieved the top CV score (CV-MSE = 0.0456, R-squared = 0.715), proving that higher-capacity models outperform GPs only when surfaces exhibit complex non-stationary transitions.
3. Diminishing Returns Near Peak Optima:
On Function 5, outputs scaled exponentially to y = 5893.21. Without target standardization and dynamic jitter tuning, acquisition scores near peak regions collapse, causing diminishing returns unless exploration bounds are actively adjusted.

---

### Question 5
**Prompt**: How might you apply hyperparameter tuning techniques to larger data sets in future rounds of the BBO capstone project submissions or more complex models in future ML/AI projects?

**Answer**:
As datasets scale beyond hundreds of observations or transition to complex deep learning models, we will evolve our hyperparameter tuning framework across three axes:

1. Automated Bayesian Optimization for Hyperparameters (AutoML):
Rather than discrete grid benchmarking, we will implement Bayesian Optimization (using Optuna or Ray Tune) to tune continuous hyperparameters—such as learning rate schedules, L2 weight penalties, network depth, and dropout rates—treating model validation error itself as a secondary black-box function.
2. Hyperband and Early Stopping:
For large-scale deep learning models, we will apply Hyperband resource allocation to terminate poorly performing hyperparameter configurations early, concentrating compute budget on promising candidates.
3. Scalable Gaussian Processes and Trust Regions (TuRBO):
For high-dimensional BBO (6D to 8D), exact GP matrix inversion scales cubically O(N^3). We will adopt Sparse Gaussian Processes (Inducing Points) and Trust Region Bayesian Optimization (TuRBO) to constrain local optimization bounds and maintain computational efficiency as datasets grow.

---

### Question 6
**Prompt**: How does tuning in this black-box set-up prepare you to think like a professional ML/AI practitioner in real-world contexts with incomplete information?

**Answer**:
Real-world industrial AI applications—such as semiconductor design, drug candidate discovery, cloud infrastructure tuning, and physical engine calibration—rarely offer clean analytical equations, unconstrained evaluation budgets, or complete state visibility.

Operating under strict black-box constraints cultivates an authentic professional practitioner mindset:
1. Empirical Evidence Over Intuition:
Tuning forces reliance on empirical cross-validation telemetry rather than subjective assumptions. When model performance degrades, practitioners trace root causes through log evidence rather than applying superficial fixes.
2. Resource-Aware Decision Making:
Recognizing that physical evaluations are expensive assets teaches practitioners to maximize decision efficiency per query, balancing local exploitation against global uncertainty reduction.
3. Robust Regularization Over Model Complexity:
Practitioners learn that simple, well-regularized models with calibrated uncertainty estimates consistently outperform complex overparameterized architectures when operating under incomplete information.
