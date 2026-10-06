# Antigravity Rules: Imperial College BBO & Deep Learning

## 1. Context First
- **Domain**: Machine Learning & AI Capstone (Imperial College London), focusing on Black-Box Optimization (BBO) across 8 benchmark functions (2D to 8D) and PyTorch Deep Learning workflows (Transformers, CNNs).
- **Environment**: Google Antigravity paired with the Gemini 3.8 Flash reasoning runtime.
- **Data State**: 10+ rounds of sequential evaluations stored as binary NumPy arrays (`function_X/initial_inputs.npy` and `initial_outputs.npy`).
- **Core Engine**: Unified master pipeline in `bbo.py` utilizing scikit-learn and PyTorch.

---

## 2. Objective Second
- **Primary Goal**: Maximize sample-efficient global optimization performance under strict query budgets (strictly 1 query per function per round).
- **Secondary Goal**: Ensure 100% deterministic reproducibility, zero token bloat, structured machine-readable logging, and robust cross-validation.
- **Architecture**: Employ multi-model 5-fold cross-validation with Automatic Relevance Determination (ARD) Gaussian Processes, hybrid Latin Hypercube/local perturbation candidate generation, and distance de-duplication.

---

## 3. Constraints Last

### 3.1 Input/Output & Data Integrity Handling
- **Submission Formatting**: All proposed queries for the portal must strictly adhere to hyphen-delimited format: `x1-x2-...-xn` formatted to exactly 6 decimal places (e.g. `0.123456-0.654321`).
- **Portal Text Rules**: Never use square brackets (`[...]`) or LaTeX math delimiters (`$...$`, `\(...\)`, `\[...\]`) in portal submission text. Use plain text or parentheses.
- **Hypercube Domain**: All input dimensions are strictly bounded to the continuous unit hypercube:
  ```
  0.000000 <= x_i <= 1.000000 for all i in [1, d]
  ```
- **Serialization**: Avoid unstructured text blobs for data steps. Every data ingestion, appending, or state export must be validated as strict JSON or binary NumPy arrays (`.npy`).

### 3.2 Robustness & Fallback Protocols (NaN / Inf Management)
- **Sanitization**: Before fitting any surrogate model (GP, MLP, Tree Ensembles), inspect `X` and `y` for `NaN`, `Inf`, or non-finite values. Filter or replace with running column median.
- **Cholesky & Covariance Regularization**:
  - If a Gaussian Process raises a `LinAlgError` or fails Cholesky decomposition, automatically increase kernel jitter/nugget:
    ```python
    alpha = max(alpha * 10, 1e-3)
    ```
  - If matrix non-positive definiteness persists, fallback to `Matern(nu=1.5)` with broader lengthscale bounds, followed by `RandomForestRegressor(n_estimators=150)` as the active surrogate.
- **Acquisition Stability**:
  - When computing Expected Improvement (EI), safeguard division by zero:
    ```python
    Z = imp / (std + 1e-12)
    ```
  - If maximum EI evaluates to `< 1e-12` across all candidates, automatically fallback to Upper Confidence Bound (UCB, `beta = 2.576`) to prevent stagnant query selection.
- **Gradient Stability (PyTorch DL)**:
  - Any gradient norm evaluating to `NaN` or exceeding `100.0` must trigger gradient clipping (`torch.nn.utils.clip_grad_norm_`).
