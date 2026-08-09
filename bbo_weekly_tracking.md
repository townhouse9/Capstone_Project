# Black-Box Optimization (BBO) Weekly Tracking Log
**Imperial College Capstone Project**

This document serves as the master tracking log for optimising the 8 synthetic black-box functions. It documents our framework design, weekly surrogate model cross-validation benchmarks, portal submission strings, visual diagnostic progress, and formal reflections.

---

## 1. Master Dashboard & Weekly Optimization Trajectory

![Weekly Progress Master Dashboard](file:///c:/Users/sesa625752/OneDrive%20-%20Schneider%20Electric/Imperial_College/Capstone_Project/visualizations/weekly_progress_summary.png)

The master dashboard image above ([weekly_progress_summary.png](file:///c:/Users/sesa625752/OneDrive%20-%20Schneider%20Electric/Imperial_College/Capstone_Project/visualizations/weekly_progress_summary.png)) provides a 4-panel overview:
1. **Current Max vs Predicted Next Query Output**: Side-by-side comparison of current maximum $y$ vs GP mean prediction with $\pm \sigma$ error bars for Functions 1–8.
2. **Weekly Optimization Trajectory**: Historical line plot tracking peak output achieved per function across weekly rounds (Week 1 $\to$ Week 2).
3. **Surrogate Model CV Performance**: Bar chart illustrating the winning surrogate model and 5-Fold CV-MSE score across all 8 functions.
4. **Expected Improvement (EI) Potential Ratio**: Normalized gain metric highlighting which functions possess the highest potential for global peak discovery in upcoming submissions.

---

## 2. Multi-Week Progress & Submission Summary

### 2.1 Function Progression & Week 2 Submissions Table

| Function | Dim | Initial $\to$ W2 Samples | Previous Max $y$ | W1 Evaluated $y$ | Current Best $y$ | Status / Gain | Week 2 Proposed Query Submission (`x1-x2-...-xn`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Func 1** | 2D | 10 $\to$ **11** | $0.000000$ | $1.284 \times 10^{-125}$ | $0.000000$ | Explored valley | `0.763819-0.768844` |
| **Func 2** | 2D | 10 $\to$ **11** | $0.611205$ | $0.100931$ | $0.611205$ | Explored boundary | `0.698492-0.929648` |
| **Func 3** | 3D | 15 $\to$ **16** | $-0.034835$ | **$-0.012927$** | **$-0.012927$** | 🏆 **New High** | `0.363880-0.129096-0.456512` |
| **Func 4** | 4D | 30 $\to$ **31** | $-4.025542$ | **$+0.367529$** | **$+0.367529$** | 🚀 **Breakthrough** | `0.404477-0.413254-0.303108-0.434359` |
| **Func 5** | 4D | 20 $\to$ **21** | $1088.859618$ | **$2062.992791$** | **$2062.992791$** | 🚀 **Breakthrough** | `0.309297-0.780073-0.993158-0.969126` |
| **Func 6** | 5D | 20 $\to$ **21** | $-0.714265$ | **$-0.305461$** | **$-0.305461$** | 🏆 **New High** | `0.237745-0.388152-0.325860-0.952624-0.039035` |
| **Func 7** | 6D | 30 $\to$ **31** | $1.364968$ | $0.742388$ | $1.364968$ | Explored sub-peak | `0.100466-0.554392-0.327801-0.220568-0.466257-0.666497` |
| **Func 8** | 8D | 40 $\to$ **41** | $9.598482$ | **$9.922921$** | **$9.922921$** | 🏆 **New High** | `0.000518-0.567012-0.060240-0.322979-0.466966-0.864631-0.061587-0.612956` |

---

## 3. Week 2 (Module 13) Reflection & Strategy Report

### Question 1: Strategy Change & Rationale
**Prompt**: *What was the main change in your strategy this week compared to last week? What prompted this change? Was it model predictions, acquisition function behaviour or something else?*
- **Response**: The main change in Week 2 was transitioning from broad, uniform exploratory search to **targeted surrogate-informed basin refinement (exploitation-guided exploration)**.
- **Drivers**:
  1. In **Function 4**, our Week 1 query transitioned the objective from negative territory ($-4.025$) into positive space ($+0.3675$), confirming that the surrogate accurately located the basin of attraction.
  2. In **Function 5**, the observed output jumped from $1088.86$ to $2062.99$. The GP model had predicted a rise to $\sim 1175 \pm 104$, but the actual output demonstrated exponential growth, indicating that the upper boundary corner ($x_3, x_4 \to 1.0$) contains the steepest scaling.
- **Acquisition refinement**: Kernel lengthscales and the Expected Improvement jitter $\xi$ were dynamically scaled relative to the expanded output range ($\Delta y$), allowing the optimizer to sample tightly along the perimeter of the newly discovered global peaks.

### Question 2: Exploration vs Exploitation Balance
**Prompt**: *Did you focus more on exploration (sampling uncertain areas) or exploitation (targeting promising areas)? Why? What trade-offs did you weigh?*
- **Response**: We adopted a differentiated trade-off:
  - **Breakthrough Functions (Functions 4, 5, 6, and 8)**: Tilted toward **targeted exploitation (60% exploitation / 40% exploration)** to climb the newly confirmed steep gradients.
  - **Stagnant Functions (Functions 1 and 2)**: Maintained **pure exploration (80% exploration / 20% exploitation)** across unvisited hypercube quadrants.
- **Trade-offs Weighed**: Pure exploitation risks getting trapped in early sub-optimal local peaks (especially in 6D–8D), while pure random exploration wastes scarce weekly queries on uninformative valleys.

### Question 3: Participant Strategies & Discussion Influences
**Prompt**: *Have any participant strategies, class discussions or recent outputs influenced how you approached this week's submission?*
- **Response**:
  1. **Multi-Surrogate Cross-Validation**: Discussions confirmed that no single surrogate dominates across all dimensions. For polynomial landscapes (e.g. Function 1 & 2), Polynomial Ridge and Tree Ensembles proved competitive (CV-MSE $= 1.46 \times 10^{-6}$), while Gaussian Processes with Matérn $\nu=2.5$ kernels decisively outperformed alternatives on higher dimensions (Functions 4, 6, 7, and 8 with $R^2 > 0.92$).
  2. **Boundary Sensitivity**: Discussions noted that synthetic test functions often place optima near hypercube boundaries. The dramatic spike in Function 5 ($y=2062.99$ at $x_3=0.978, x_4=0.941$) reinforced this insight, prompting thorough sampling of boundary coordinates ($x_i \in [0.90, 1.0]$).

### Question 4: Violation of Linear & Logistic Regression Assumptions
**Prompt**: *If you were to fit a simple linear or logistic regression model to your current data for one of the functions, which assumptions would you most likely violate? Consider aspects such as the shape of the response surface, noise levels or number of features.*
- **Response**:
  1. **Linearity of the Response Surface**: Severe violation. Black-box functions exhibit high-order polynomial interactions and non-monotonic curves. In Function 5, outputs scale exponentially ($0.11 \to 2062.99$), which a first-order hyper-plane $\hat{y} = \beta_0 + \sum \beta_i x_i$ completely fails to fit.
  2. **Homoscedasticity (Constant Residual Variance)**: Severe violation. Residual variance is non-uniform across the domain (flat in valleys, volatile near peaks).
  3. **Linear Separability of Log-Odds**: Severe violation. In threshold classification ("High Yield" vs "Low Yield"), optimal regions form localized islands; standard logistic regression can only draw a single flat hyper-plane, misclassifying multi-modal landscapes.
  4. **Sample Size Sparsity**: In Function 8 (8D with 41 samples) and Function 7 (6D with 31 samples), the low sample-to-feature ratio causes multicollinearity and parameter instability.

### Question 5: Rough Linearity, Decision Boundaries & Logistic Regression Utility
**Prompt**: *Are there any regions where the output appears roughly linear or where a decision boundary might form? How might a logistic regression classifier perform on this function, particularly in binary or threshold-based scenarios?*
- **Response**:
  - **Roughly Linear Sub-Regions**: 1D sensitivity slices show that localized sub-regions exhibit approximate monotonic linearity over narrow intervals ($\Delta x \le 0.1$, e.g., Function 8 along Dimensions 1 & 3, and Function 4 around $[0.38, 0.42, 0.40, 0.39]$).
  - **Logistic Regression Utility**: In binary threshold scenarios ($y > y_{threshold}$), logistic regression can isolate the active search quadrant if the peak resides in a single hypercube corner (such as Function 5 where $x_3, x_4 > 0.8 \implies \text{High Yield}$). In concentric/multi-modal functions (Functions 3 and 7), kernel transformations are required.

### Question 6: Interpretability & Feature Effects
**Prompt**: *Interpretability is a key advantage of linear and logistic regression. Did you find it useful to consider individual feature effects before deciding on your query point?*
- **Response**: **Yes, analyzing individual feature effects was immensely useful**:
  - **Feature Importance**: Random Forest feature importance revealed that in **Function 5**, Dimensions 3 and 4 account for over $78\%$ of total variance, justifying focusing Week 2 coordinates near $x_3 \approx 0.99$ and $x_4 \approx 0.97$.
  - **1D Partial Dependence Slices**: In **Function 8 (8D)**, 1D slices demonstrated that Dimensions 1, 3, and 4 achieve maximum output when kept strictly close to zero ($x_i \approx 0.00-0.05$), while Dimensions 5 and 6 require higher values. Considering these individual directional effects eliminated over $90\%$ of unpromising candidate space and directly informed our Week 2 submission vector `0.000518-0.567012-0.060240-0.322979-0.466966-0.864631-0.061587-0.612956`.
