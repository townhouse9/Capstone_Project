# Black-Box Optimization (BBO) Weekly Tracking Log
**Imperial College Capstone Project**

This document serves as the master tracking log for optimising the 8 synthetic black-box functions. It documents the framework design, weekly surrogate model cross-validation benchmarks, portal submission strings, visual diagnostic progress, and formal reflections.

---

## 1. Initial System Design & Methodological Setup

### 1.1 Architecture & Pipeline Design (`bbo_pipeline.py`)
To handle weekly data expansion seamlessly as new output evaluations $y$ become available, we designed an automated Python workflow:
- **Data Ingestion**: Loads `.npy` files dynamically across `function_1` through `function_8`.
- **Surrogate Modeling Suite**: Benchmarks probabilistic Bayesian Optimization (Gaussian Process with Matérn $\nu=2.5$, $\nu=1.5$, and RBF kernels) against non-probabilistic ML Regression surrogates (**Random Forest**, **Extra Trees**, **Polynomial Ridge**, and **Gradient Boosting**).
- **Cross-Validation Framework**: Evaluates all candidate surrogate models per function using K-Fold Cross-Validation (CV Mean Squared Error and $R^2$).
- **Acquisition & Candidate Search**:
  - *2D Functions*: Dense evaluation over a 200x200 uniform mesh grid ($40,000$ points).
  - *3D–8D Functions*: Monte Carlo / Latin Hypercube sampling over $30,000$ candidate points in $[0, 1]^d$.
  - *Acquisition Function*: **Expected Improvement (EI)** with dynamic relative jitter $\xi = 0.01 \times \Delta y$, with fallback to **Upper Confidence Bound (UCB)** ($\beta=2.576$).
- **Strict Format Enforcer**: Automatically formats query vectors into the required submission string: `x1-x2-x3-...-xn` (where each coordinate starts with `0.` and is rounded to six decimal places).
- **Automated Summary Dashboard Generator**: Automatically generates and updates [visualisations/weekly_progress_summary.png](file:///c:/Users/sesa625752/OneDrive%20-%20Schneider%20Electric/Imperial_College/Capstone_Project/visualizations/weekly_progress_summary.png) tracking current max vs predicted peak, multi-week optimisation trajectories, surrogate CV model rankings, and expected improvement ratios.

---

## 2. Master Dashboard & Weekly Trajectory Visualisation

![Weekly Progress Master Dashboard](file:///c:/Users/sesa625752/OneDrive%20-%20Schneider%20Electric/Imperial_College/Capstone_Project/visualizations/weekly_progress_summary.png)

The master dashboard image above ([weekly_progress_summary.png](file:///c:/Users/sesa625752/OneDrive%20-%20Schneider%20Electric/Imperial_College/Capstone_Project/visualizations/weekly_progress_summary.png)) provides a 4-panel overview:
1. **Current Max vs Predicted Next Query Output**: Side-by-side comparison of current maximum $y$ vs GP mean prediction with $\pm \sigma$ error bars for Functions 1–8.
2. **Weekly Optimization Trajectory**: Historical line plot tracking peak output achieved per function across weekly rounds (Week 1, Week 2, ...).
3. **Surrogate Model CV Performance**: Bar chart illustrating the winning surrogate model and 5-Fold CV-MSE score across all 8 functions.
4. **Expected Improvement (EI) Potential Ratio**: Normalized gain metric highlighting which functions possess the highest potential for global peak discovery in upcoming submissions.

---

## 3. Week 1 (Module 12) Function Analysis & Query Selection

### 3.1 Function Summary Table

| Function | Dim | Initial Samples | Current Max $y$ | Best Model (by CV-MSE) | Predicted $y$ (GP Mean $\pm$ Std) | Formatted Portal Submission String |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Func 1** | 2D | 10 | $0.000000$ | GP (Matérn 1.5) | $-0.000361 \pm 0.001082$ | `0.829146-0.256281` |
| **Func 2** | 2D | 10 | $0.611205$ | Gradient Boosting | $0.590832 \pm 0.134885$ | `0.773869-0.944724` |
| **Func 3** | 3D | 15 | $-0.034835$ | GP (Matérn 1.5) | $-0.042633 \pm 0.066021$ | `0.375439-0.360518-0.460524` |
| **Func 4** | 4D | 30 | $-4.025542$ | GP (Matérn 2.5) | $-1.907575 \pm 0.783341$ | `0.384555-0.428957-0.409752-0.392875` |
| **Func 5** | 4D | 20 | $1088.859618$ | Polynomial Ridge | $1175.292123 \pm 104.756388$ | `0.314984-0.829694-0.978179-0.941555` |
| **Func 6** | 5D | 20 | $-0.714265$ | GP (Matérn 2.5) | $-0.525263 \pm 0.236127$ | `0.450383-0.325133-0.471110-0.818738-0.124614` |
| **Func 7** | 6D | 30 | $1.364968$ | GP (RBF) | $1.116168 \pm 0.188577$ | `0.038644-0.565902-0.200304-0.089878-0.396909-0.792838` |
| **Func 8** | 8D | 40 | $9.598482$ | GP (Matérn 1.5) | $10.188564 \pm 0.364543$ | `0.050401-0.196416-0.014677-0.026846-0.870052-0.391372-0.184226-0.556620` |

---

## 4. Detailed Function-by-Function Diagnostics

### Function 1 (2D)
- **Current Max**: $0.000000$ (7.71e-16) at $[0.731024, 0.733000]$
- **Model Assessment**: GP with Matérn 1.5 kernel achieved the lowest CV-MSE ($1.55 \times 10^{-6}$). Polynomial Ridge and GP Matérn 2.5 also performed competitively.
- **Acquisition Strategy**: Due to tiny response values ($y \sim 10^{-16}$), acquisition jitter $\xi$ was dynamically scaled relative to sample output range. The acquisition maximum targets high-uncertainty regions with potential global peak near `0.829146-0.256281`.
- **Visualisation**: `visualizations/function_1_week1.png`

### Function 2 (2D)
- **Current Max**: $0.611205$ at $[0.702637, 0.926564]$
- **Model Assessment**: Ensemble models (Gradient Boosting & Random Forest) provided strong localized fitting, while GP Matérn 2.5 modeled smooth spatial transitions.
- **Acquisition Strategy**: Expected Improvement (EI) highlights a promising peak near the upper boundary $[0.773869, 0.944724]$.
- **Visualisation**: `visualizations/function_2_week1.png`

### Function 3 (3D)
- **Current Max**: $-0.034835$ at $[0.492581, 0.611593, 0.340176]$
- **Model Assessment**: GP Matérn 1.5 outperformed tree-based models with CV-MSE of $0.008429$.
- **Acquisition Strategy**: EI query point `0.375439-0.360518-0.460524` explores the central region where variance remains high.
- **Visualisation**: `visualizations/function_3_week1.png`

### Function 4 (4D)
- **Current Max**: $-4.025542$ at $[0.577766, 0.428772, 0.425826, 0.249007]$
- **Model Assessment**: GP Matérn 2.5 demonstrated exceptional performance ($R^2 = 0.9387$, CV-MSE $= 2.749$), significantly outperforming Random Forest ($R^2 = 0.574$).
- **Acquisition Strategy**: Recommended query `0.384555-0.428957-0.409752-0.392875` balances exploration around the high-performing 4D basin.
- **Visualisation**: `visualizations/function_4_week1.png`

### Function 5 (4D)
- **Current Max**: $1088.859618$ at $[0.224189, 0.846480, 0.879484, 0.878516]$
- **Model Assessment**: Extremely high variance response (range $0.11$ to $1088.86$). Polynomial Ridge and Extra Trees captured global quadratic gradients effectively.
- **Acquisition Strategy**: Proposed query `0.314984-0.829694-0.978179-0.941555` explores the upper corner region where function growth is exponential.
- **Visualisation**: `visualizations/function_5_week1.png`

### Function 6 (5D)
- **Current Max**: $-0.714265$ at $[0.728186, 0.154693, 0.732552, 0.693996, 0.056401]$
- **Model Assessment**: GP Matérn 2.5 yielded top CV score ($0.118070$).
- **Acquisition Strategy**: Selected query `0.450383-0.325133-0.471110-0.818738-0.124614` explores unsaturated dimensions.
- **Visualisation**: `visualizations/function_6_week1.png`

### Function 7 (6D)
- **Current Max**: $1.364968$ at $[0.057896, 0.491672, 0.247422, 0.218118, 0.420428, 0.730970]$
- **Model Assessment**: GP RBF provided the smoothest surrogate fit (CV-MSE $= 0.085345$).
- **Acquisition Strategy**: EI query `0.038644-0.565902-0.200304-0.089878-0.396909-0.792838` samples near the boundary of the highest recorded peak.
- **Visualisation**: `visualizations/function_7_week1.png`

### Function 8 (8D)
- **Current Max**: $9.598482$ at $[0.056447, 0.065956, 0.022929, 0.038786, 0.403935, 0.801055, 0.488307, 0.893085]$
- **Model Assessment**: Excellent model fit across all GP variants ($R^2 > 0.90$) and Polynomial Ridge ($R^2 = 0.8947$), outperforming Random Forest ($R^2 = 0.4395$).
- **Acquisition Strategy**: EI maximum predicts an output of $10.188564 \pm 0.364543$ at coordinate `0.050401-0.196416-0.014677-0.026846-0.870052-0.391372-0.184226-0.556620`.
- **Visualisation**: `visualizations/function_8_week1.png`

---

## 5. Week 1 Reflection & Strategy Report

### Question 1: What was the main principle or heuristic used to decide on each query point?
Our primary heuristic was balancing **exploration of high-epistemic-uncertainty regions** with **exploitation of known high-output regions** using **Expected Improvement (EI)** derived from Gaussian Process (GP) surrogates. 
For low-dimensional functions (Functions 1 & 2), grid search over GP mean and uncertainty maps revealed distinct local optima. For higher-dimensional functions (Functions 4, 7, and 8), GP surrogates demonstrated superior predictive capability ($R^2 > 0.90$), enabling EI to guide queries toward regions with high predicted mean and significant posterior variance.

### Question 2: Which function(s) were most challenging to query, and why?
- **Function 5 (4D)**: Displayed an extreme output range ($0.11$ to $1088.86$), causing tree-based regressors to struggle with extrapolation beyond boundary training samples.
- **Function 1 (2D)**: Had extremely small magnitude outputs ($y \approx 10^{-16}$ to $-0.0036$). Standard static acquisition parameters ($\xi = 0.01$) led to flat EI landscapes. Scaling $\xi$ relative to the empirical range $\Delta y$ was essential to restore meaningful acquisition gradients.
- **Function 8 (8D)**: High dimensionality (8D with only 40 initial samples) creates severe volume sparsity. While the GP model fit well ($R^2 = 0.9089$), random candidate sampling requires tens of thousands of points to adequately cover the domain.

### Question 3: How do you plan to adjust your strategy in future rounds?
1. **Dynamic Model Ensembling**: As sample sizes grow from 10–40 up to 22–50+, we will weight surrogate models dynamically based on their weekly CV scores. If non-linear tree ensembles (Random Forest / Gradient Boosting) improve relative to GPs on complex functions like Function 5, we will incorporate optimistic upper bounds from tree variance.
2. **Hyperparameter Adaptation**: We will update kernel lengthscale bounds dynamically as sample density increases to capture sharper local features.
3. **Exploitation Shift**: In early rounds (Modules 12–15), we prioritize exploration ($\xi = 0.01$). In later rounds (Modules 20–24), we will gradually reduce $\xi$ to exploit discovered global peaks.
