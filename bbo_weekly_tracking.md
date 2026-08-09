# Black-Box Optimization (BBO) Weekly Tracking Log
**Imperial College Capstone Project**

This document serves as the master tracking log for optimising the 8 synthetic black-box functions. It documents our framework design, weekly surrogate model cross-validation benchmarks, portal submission strings, visual diagnostic progress, and formal reflections across all rounds.

---

## 1. Master Dashboard & Weekly Optimization Trajectory

![Weekly Progress Master Dashboard](file:///c:/Users/sesa625752/OneDrive%20-%20Schneider%20Electric/Imperial_College/Capstone_Project/visualizations/weekly_progress_summary.png)

The master dashboard image above ([weekly_progress_summary.png](file:///c:/Users/sesa625752/OneDrive%20-%20Schneider%20Electric/Imperial_College/Capstone_Project/visualizations/weekly_progress_summary.png)) provides a 4-panel overview:
1. **Current Max vs Predicted Next Query Output**: Side-by-side comparison of current maximum $y$ vs GP mean prediction with $\pm \sigma$ error bars for Functions 1–8.
2. **Weekly Optimization Trajectory**: Historical line plot tracking peak output achieved per function across weekly rounds (Week 1 $\to$ Week 2 $\to$ Week 3).
3. **Surrogate Model CV Performance**: Bar chart illustrating the winning surrogate model and 5-Fold CV-MSE score across all 8 functions.
4. **Expected Improvement (EI) Potential Ratio**: Normalized gain metric highlighting which functions possess the highest potential for global peak discovery in upcoming submissions.

---

## 2. Multi-Week Progress & Week 3 Submissions Summary

### 2.1 Function Progression & Week 3 Submissions Table

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

## 3. Week 3 (Module 14) Reflection & Strategy Report

### Question 1: Strategy Change & Hyperparameter Tuning
**Prompt**: *How has your query strategy changed from earlier rounds? Do you rely more on model predictions, or are you still exploring new regions? Do you tune hyperparameters or rely on heuristics?*
- **Response**:
  1. **Evolution of Strategy**: In Round 1, our strategy was purely global exploration. By Round 3, we have shifted to a **hybrid model-driven acquisition regime**:
     - For functions where a confirmed basin has been established (**Function 4, Function 5, Function 8**), we rely heavily on surrogate model mean predictions $\mu(x)$ and fine-grained Expected Improvement (EI) to climb local peaks.
     - For multi-modal functions with flat feedback (**Function 1 and Function 3**), we reserve acquisition weight for unvisited sub-spaces (high $\sigma(x)$).
  2. **Hyperparameter Tuning**: We perform **Maximum Marginal Likelihood (MML) optimization** with 15 restarts for Gaussian Process hyperparameters ($\ell, \sigma_f^2, \sigma_n^2$) combined with 5-Fold Cross-Validation model selection, while dynamically tuning jitter $\xi$ to match the empirical dynamic range $\Delta y$.

### Question 2: Balancing Exploration vs Exploitation
**Prompt**: *How do you balance exploration against exploitation? Do you focus more on areas known to perform well, or are you still sampling from untested regions?*
- **Response**: We enforce a **context-aware exploration/exploitation balance**:
  - **Exploitation-Dominant Functions (Function 5 & Function 2)**: On Function 5, consecutive queries ($y = 1088 \to 2062 \to 2178$) proved that the boundary region ($x_2 \approx 0.80, x_3 \approx 0.97, x_4 \approx 0.98$) contains a massive ascending ridge, concentrating $70\%$ of sampling budget on local exploitation.
  - **Exploration-Dominant Functions (Function 3 & Function 1)**: On Function 3, Week 2 returned $y = -0.0337$, prompting the optimizer to target an untested corner of the 3D cube (`0.653656-0.956221-0.003021`), where GP variance $\sigma(x)$ was highest.

### Question 3: Support Vector Machines (SVMs) & Region Classification
**Prompt**: *How would SVMs change your approach? Could you use a soft-margin SVM to classify high vs low performance regions? Would a kernel SVM help if the response surface is non-linear?*
- **Response**:
  1. **Region Classification**: By setting an output threshold $\tau$ (e.g., top 20th percentile $y_i \ge y_{80\%}$), continuous regression can be cast as a binary classification problem: $z_i = +1$ (High-Yield Zone) vs $z_i = -1$ (Sub-Optimal Valley).
  2. **Soft-Margin SVM ($C$-SVM)**: Introduces slack variables $\xi_i \ge 0$ to handle boundary overlap and noise. The margin band separates viable regions from dead zones, allowing candidate samplers to filter out unpromising hypercube volume.
  3. **Kernel SVMs (RBF Kernels)**: Linear SVM hyperplanes fail on non-linear multi-modal surfaces. An **RBF Kernel SVM** maps inputs into an infinite-dimensional feature space, constructing closed non-linear decision boundaries around isolated peaks.

### Question 4: Model Limitations & Dimensional Relevance
**Prompt**: *What limitations of your current model become apparent as data grows? Is it overfitting? Do any features or dimensions emerge as irrelevant?*
- **Response**:
  1. **Computational Inversion Limits**: Exact Gaussian Process regression scales as $\mathcal{O}(N^3)$. As data points grow into hundreds, sparse approximations (FITC, Inducing Points) become necessary.
  2. **Stationarity Assumption**: Standard Matérn/RBF kernels assume uniform lengthscale $\ell_d$ everywhere. In **Function 5**, the surface is flat in lower quadrants but ascends exponentially near the top corner; stationary GPs struggle with such heteroscedasticity.
  3. **Dimensional Irrelevance**:
     - In **Function 5 (4D)**, Dimensions 3 and 4 dominate ($>75\%$ importance), while Dimension 1 behaves almost as an irrelevant nuisance parameter.
     - In **Function 8 (8D)**, Dimensions 1, 3, and 7 exhibit extreme sensitivity near zero, while Dimensions 2 and 8 have broad, gentle curvatures.

### Question 5: Data Science Mindset & Incomplete Knowledge
**Prompt**: *How does this black-box set-up prepare you to think like a data scientist when faced with incomplete knowledge in other projects?*
- **Response**:
  1. **Decision Making Under Strict Budgets**: In real-world ML and industry (A/B testing, drug screening, hyperparameter tuning), observations are expensive. This setup enforces **principled probabilistic modeling** over brute-force search.
  2. **Quantifying Epistemic Uncertainty**: Rather than relying only on predictions $\hat{y}$, data scientists must model variance $\sigma(x)$. Knowing where the model is ignorant is as vital as knowing where it is confident.
  3. **Agile Iterative Reasoning**: Each round functions as a closed-loop experiment: hypothesize $\to$ query $\to$ evaluate feedback $\to$ diagnose model errors $\to$ adapt strategy.
