# Black-Box Optimization (BBO) - Week 3 (Module 14) Progress & Strategy Report
**Imperial College Capstone Project**

---

## 1. Week 3 Input Queries & Output Evaluation Summary

With the addition of Week 3 evaluations ($N = 12$ to $42$ samples across the 8 functions), our Bayesian Optimization and Surrogate Modeling framework achieved another **all-time high record maximum on Function 5 ($y = 2178.4478$)**, and confirmed strong peak clustering on Functions 2, 4, 7, and 8.

### 1.1 Progression Table (Week 1 $\to$ Week 3)

| Function | Dim | Initial $\to$ Current Samples | Previous Best $y$ | Week 2 Evaluated $y$ | Current Best $y$ | Status / Gain | Week 3 Proposed Query Submission (`x1-x2-...-xn`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Func 1** | 2D | 10 $\to$ **12** | $7.71 \times 10^{-16}$ | $4.056 \times 10^{-27}$ | $7.71 \times 10^{-16}$ | Constrained near peak | `0.793970-0.708543` |
| **Func 2** | 2D | 10 $\to$ **12** | $0.611205$ | $0.595225$ | $0.611205$ | Exploited high plateau | `0.698492-0.909548` |
| **Func 3** | 3D | 15 $\to$ **17** | $-0.012927$ | $-0.033750$ | $-0.012927$ | Explored mode 2 | `0.653656-0.956221-0.003021` |
| **Func 4** | 4D | 30 $\to$ **32** | $+0.367529$ | $-0.715815$ | **$+0.367529$** | Boundary mapping | `0.357349-0.426936-0.472854-0.502713` |
| **Func 5** | 4D | 20 $\to$ **22** | $2062.992791$ | **$2178.447793$** | **$2178.447793$** | 🚀 **New Record High** | `0.448615-0.806871-0.971916-0.980875` |
| **Func 6** | 5D | 20 $\to$ **22** | $-0.305461$ | $-0.753035$ | $-0.305461$ | Explored sub-space | `0.374454-0.282104-0.644553-0.708365-0.016968` |
| **Func 7** | 6D | 30 $\to$ **32** | $1.364968$ | $1.139851$ | $1.364968$ | High yield neighborhood | `0.084725-0.411102-0.266850-0.290702-0.452889-0.815320` |
| **Func 8** | 8D | 40 $\to$ **42** | $9.922921$ | $9.535042$ | **$9.922921$** | High plateau exploration | `0.122085-0.080745-0.011016-0.224789-0.742507-0.480219-0.165680-0.248194` |

---

## 2. Updated Master Dashboard & Weekly Trajectories

![Weekly Progress Master Dashboard](file:///c:/Users/sesa625752/OneDrive%20-%20Schneider%20Electric/Imperial_College/Capstone_Project/visualizations/weekly_progress_summary.png)

---

## 3. Answers to Weekly Reflection Questions

### Prompt 1: How has your query strategy changed from earlier rounds? Do you rely more on model predictions, or are you still exploring new regions? Do you tune hyperparameters or rely on heuristics?

**Response**:
1. **Evolution of Strategy**: In Round 1, our strategy was purely global exploration (mapping overall landscape topology and epistemic uncertainty). By Round 3, we have shifted to a **hybrid model-driven acquisition regime**:
   - For functions where a confirmed basin has been established (**Function 4, Function 5, Function 8**), we rely heavily on surrogate model mean predictions $\mu(x)$ and fine-grained Expected Improvement (EI) to systematically climb local peaks.
   - For multi-modal functions with flat or ambiguous feedback (**Function 1 and Function 3**), we reserve acquisition weight for unvisited sub-spaces (high $\sigma(x)$).
2. **Hyperparameter Tuning vs. Heuristics**:
   - Rather than relying on static heuristic choices (such as fixed lengthscales or arbitrary search grids), we now execute **rigorous Maximum Marginal Likelihood (MML) optimization** with 15 random restarts for Gaussian Process hyperparameters ($\ell, \sigma_f^2, \sigma_n^2$) combined with 5-Fold Cross-Validation model selection.
   - We dynamically tune the exploration jitter $\xi$ to match the empirical dynamic range ($\Delta y$) of each function, ensuring that numerical rounding never collapses acquisition gradients.

---

### Prompt 2: How do you balance exploration against exploitation? Do you focus more on areas known to perform well, or are you still sampling from untested regions?

**Response**:
We enforce a **context-aware exploration/exploitation balance** governed by the confidence and curvature of the surrogate:
- **Exploitation-Dominant Functions (Function 5 & Function 2)**: 
  - On Function 5, our consecutive queries ($y = 1088 \to 2062 \to 2178$) proved that the boundary region ($x_2 \approx 0.80, x_3 \approx 0.97, x_4 \approx 0.98$) contains a massive ascending ridge. We concentrate $70\%$ of sampling budget on local exploitation around this ridge while allowing $30\%$ exploration on Dimension 1 ($x_1$).
- **Exploration-Dominant Functions (Function 3 & Function 1)**:
  - On Function 3, Week 2's query at $[0.36, 0.13, 0.46]$ returned $y = -0.0337$, indicating that the central basin is relatively flat. For Week 3, the optimizer targeted a completely untested corner of the 3D cube (`0.653656-0.956221-0.003021`), where GP posterior variance $\sigma(x)$ was highest.
- **The Core Trade-Off**: Sampling untested regions prevents catastrophic convergence to suboptimal local modes (especially in 6D–8D), while sampling near proven high-output zones accelerates convergence when a steep gradient is unlocked.

---

### Prompt 3: How would SVMs change your approach? Could you use a soft-margin SVM to classify high vs low performance regions? Would a kernel SVM help if the response surface is non-linear?

**Response**:
1. **Reformulating Optimization as Region Classification**:
   - By setting a threshold $\tau$ (e.g., the top 20th percentile of observed outputs, $y_i \ge y_{80\%}$), we can transform continuous regression into a binary classification problem: $z_i = +1$ (High-Yield Zone) vs $z_i = -1$ (Sub-Optimal Valley).
2. **Soft-Margin SVM for Noisy / Overlapping Regions**:
   - A **Soft-Margin SVM ($C$-SVM)** would introduce slack variables $\xi_i \ge 0$, allowing for boundary overlap and observation noise. The regularization parameter $C$ controls the penalty for misclassifying borderline points. This would define a margin band separating "safe exploration zones" from "low-yield dead zones", allowing candidate samplers to filter out unpromising hypercube volume prior to GP acquisition.
3. **Power of Kernel SVMs (RBF / Polynomial Kernels)**:
   - Because our black-box functions exhibit non-linear multi-modality, linear SVM hyperplanes would fail completely. An **RBF Kernel SVM** ($\Phi(x)$) maps the input coordinates into an infinite-dimensional feature space, enabling the construction of complex, non-linear, closed decision contours (hyperspherical or multi-island boundaries) enclosing isolated local peaks. This would serve as a fast region-filtering surrogate.

---

### Prompt 4: What limitations of your current model become apparent as data grows? Is it overfitting? Do any features or dimensions emerge as irrelevant?

**Response**:
1. **Computational & Inversion Scaling**:
   - Gaussian Process regression requires inverting the covariance matrix $\mathbf{K} \in \mathbb{R}^{N \times N}$, which scales with computational complexity $\mathcal{O}(N^3)$. While trivial at $N=42$, as data points grow into hundreds, exact GP fitting requires sparse approximations (e.g., FITC, Inducing Points).
2. **Stationarity Assumption Limits**:
   - Standard Matérn/RBF kernels assume **stationarity** (the same lengthscale $\ell_d$ applies uniformly everywhere across $[0, 1]^d$). In **Function 5**, the surface is almost flat in the lower quadrants but ascends exponentially near $[0.4, 0.8, 0.98, 0.98]$. Stationary GPs struggle with such non-stationary heteroscedasticity, occasionally underestimating peak heights.
3. **Dimensional Irrelevance & Sparsity Detection**:
   - Using Automatic Relevance Determination (ARD) and Random Forest feature importances, we observed clear dimensional hierarchies:
     - In **Function 5 (4D)**, Dimensions 3 and 4 dominate ($>75\%$ importance), while Dimension 1 behaves almost as an irrelevant nuisance parameter.
     - In **Function 8 (8D)**, Dimensions 1, 3, and 7 exhibit extreme sensitivity near zero, while Dimensions 2 and 8 exhibit relatively gentle, broad curvatures.

---

### Prompt 5: How does this black-box set-up prepare you to think like a data scientist when faced with incomplete knowledge in other projects?

**Response**:
This black-box optimization challenge mirrors the most challenging real-world data science and industrial engineering scenarios (e.g., hyperparameter tuning in deep networks, drug discovery screening, chemical process optimization, A/B testing):
1. **Decision Making Under Strict Budget Constraints**:
   - In real-world ML and industry, collecting labels or running physical tests is expensive, time-consuming, or hazardous. This setup forces a data scientist to abandon brute-force random grid searches in favor of **principled probabilistic surrogate modeling** where every single sample point must be mathematically justified.
2. **Quantifying Epistemic Uncertainty**:
   - Rather than relying solely on point predictions $\hat{y}$, data scientists must model uncertainty $\sigma(x)$. Knowing *where the model is ignorant* is just as important as knowing where it is confident.
3. **Agile Iteration & Hypothesis Testing**:
   - Each round functions as a closed-loop experiment: hypothesize $\to$ query $\to$ evaluate feedback $\to$ diagnose model errors $\to$ adapt strategy. This builds the fundamental discipline of iterative reasoning required for complex machine learning projects.
