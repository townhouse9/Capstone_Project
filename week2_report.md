# Black-Box Optimization (BBO) - Week 2 (Module 13) Progress & Strategy Report
**Imperial College Capstone Project**

---

## 1. Week 2 Input Queries & Output Evaluation Summary

Following the submission of our Week 1 queries, we evaluated the newly received output values ($y$). The optimization framework achieved remarkable progress, setting **new all-time high global maximums on 5 out of the 8 functions** (Functions 3, 4, 5, 6, and 8).

### 1.1 Progression Table (Week 1 $\to$ Week 2)

| Function | Dim | Initial Samples $\to$ Total | Previous Best $y$ | Week 1 Evaluated $y$ | Status / Gain | Week 2 Proposed Query Submission (`x1-x2-...-xn`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Func 1** | 2D | 10 $\to$ **11** | $0.000000$ | $1.284 \times 10^{-125}$ | Explored valley | `0.763819-0.768844` |
| **Func 2** | 2D | 10 $\to$ **11** | $0.611205$ | $0.100931$ | Explored boundary | `0.698492-0.929648` |
| **Func 3** | 3D | 15 $\to$ **16** | $-0.034835$ | **$-0.012927$** | 🏆 **New All-Time High** | `0.363880-0.129096-0.456512` |
| **Func 4** | 4D | 30 $\to$ **31** | $-4.025542$ | **$+0.367529$** | 🚀 **Massive Breakthrough** | `0.404477-0.413254-0.303108-0.434359` |
| **Func 5** | 4D | 20 $\to$ **21** | $1088.859618$ | **$2062.992791$** | 🚀 **Massive Breakthrough** | `0.309297-0.780073-0.993158-0.969126` |
| **Func 6** | 5D | 20 $\to$ **21** | $-0.714265$ | **$-0.305461$** | 🏆 **New All-Time High** | `0.237745-0.388152-0.325860-0.952624-0.039035` |
| **Func 7** | 6D | 30 $\to$ **31** | $1.364968$ | $0.742388$ | Explored sub-peak | `0.100466-0.554392-0.327801-0.220568-0.466257-0.666497` |
| **Func 8** | 8D | 40 $\to$ **41** | $9.598482$ | **$9.922921$** | 🏆 **New All-Time High** | `0.000518-0.567012-0.060240-0.322979-0.466966-0.864631-0.061587-0.612956` |

---

## 2. Master Dashboard & Weekly Trajectory Visualisation

![Weekly Progress Master Dashboard](file:///c:/Users/sesa625752/OneDrive%20-%20Schneider%20Electric/Imperial_College/Capstone_Project/visualizations/weekly_progress_summary.png)

The master dashboard above tracks:
1. **Current Max vs Predicted Next Peak**: Comparison with prediction uncertainty bounds $\pm \sigma$.
2. **Weekly Trajectory**: Historical progression across weekly iterations showing steep upward trajectories for Functions 4, 5, and 8.
3. **Surrogate Model CV Benchmark**: 5-Fold Cross-Validation error and winning surrogate model type across functions.
4. **Expected Improvement (EI) Potential**: Normalized expected gain informing query selection.

---

## 3. Weekly Reflection & Strategy Answers

### Prompt 1: What was the main change in your strategy this week compared to last week? What prompted this change? Was it model predictions, acquisition function behaviour or something else?

**Response**:
The main strategic change in Week 2 was transitioning from broad, uniform domain exploration to **targeted, surrogate-informed basin refinement (exploitation-guided exploration)**:
- **What prompted the change**: The newly returned output evaluations provided decisive signals regarding the underlying landscapes:
  1. In **Function 4**, our Week 1 query transitioned the objective from a negative valley ($-4.025$) into positive territory ($+0.3675$), confirming that the surrogate had successfully located the basin of attraction.
  2. In **Function 5**, the observed output jumped from $1088.86$ to $2062.99$. The GP model had predicted a rise to $\sim 1175 \pm 104$, but the actual output demonstrated super-linear growth, signaling that the upper boundary corner ($x_3, x_4 \to 1.0$) contains exponential scaling.
- **Acquisition Function Adjustment**: We updated kernel lengthscale bounds dynamically and scaled the Expected Improvement jitter $\xi$ relative to the newly expanded dynamic range ($\Delta y$). This prevented acquisition saturation and allowed the optimizer to tightly sample the perimeter of the newly discovered global peaks.

---

### Prompt 2: Did you focus more on exploration (sampling uncertain areas) or exploitation (targeting promising areas)? Why? What trade-offs did you weigh?

**Response**:
We employed a **differentiated exploration/exploitation trade-off** based on function performance:
- **For Breakthrough Functions (Functions 4, 5, 6, and 8)**: We tilted the balance toward **targeted exploitation (60% exploitation / 40% exploration)**. Because these functions set substantial new all-time highs, our priority was sampling near the high-gradient neighborhood to climb the discovered peak before it flattens.
- **For Stagnant/Valley Functions (Functions 1 and 2)**: We maintained **pure exploration (80% exploration / 20% exploitation)**. In Function 1, our Week 1 sample ($y = 1.28 \times 10^{-125}$) confirmed that the $[0.829, 0.256]$ region was an uninformative trough; hence, Week 2 shifts query coordinates to unvisited quadrants with high epistemic variance ($\sigma(x)$).
- **Trade-offs Weighed**:
  - *Risk of premature exploitation*: Getting trapped in a mediocre local peak in high-dimensional space (e.g., 6D–8D).
  - *Risk of excessive exploration*: Wasting scarce weekly queries sampling far-away regions when a confirmed steep gradient has already been unlocked.

---

### Prompt 3: Have any participant strategies, class discussions or recent outputs influenced how you approached this week's submission?

**Response**:
Yes, multiple key insights from class discussions and recent batch outputs shaped this week's approach:
1. **Multi-Surrogate Cross-Validation**: Discussion emphasized that no single surrogate family dominates across all dimensions. For low-dimensional and polynomial landscapes (e.g., Function 1 and Function 2), Polynomial Ridge and Tree Ensembles proved competitive (CV-MSE $= 1.46 \times 10^{-6}$), while Gaussian Processes with Matérn $\nu=2.5$ kernels decisively outperformed all alternatives on higher dimensions (Functions 4, 6, 7, and 8 with $R^2 > 0.92$).
2. **Boundary Sensitivity in Synthetic Functions**: Class discussions noted that synthetic test functions often place optima near hypercube boundaries. The dramatic spike in Function 5 ($y=2062.99$ at $x_3=0.978, x_4=0.941$) reinforced this insight, prompting our Week 2 candidate sampler to thoroughly evaluate boundary coordinates ($x_i \in [0.90, 1.0]$).

---

### Prompt 4: If you were to fit a simple linear or logistic regression model to your current data for one of the functions, which assumptions would you most likely violate? Consider aspects such as the shape of the response surface, noise levels or number of features.

**Response**:
If fitting simple linear or logistic regression models to these black-box functions, several fundamental statistical assumptions would be severely violated:

1. **Linearity of the Response Surface (Linear Regression)**:
   - *Severe Violation*: The true underlying response surfaces are highly non-linear, multi-modal, and non-monotonic. In **Function 5**, output values scale exponentially from $0.11$ to $2062.99$, indicating strong multi-variable interaction terms ($x_1 \cdot x_2 \cdot x_3$) that a first-order hyper-plane $\hat{y} = \beta_0 + \sum \beta_i x_i$ completely fails to capture.
2. **Homoscedasticity (Constant Residual Variance)**:
   - *Severe Violation*: The residual variance is non-uniform across the domain. Valleys exhibit flat, near-zero gradients with small variances, whereas regions near peaks display extreme output volatility. Linear regression assumes $\text{Var}(\epsilon_i) = \sigma^2$ everywhere.
3. **Linear Separability of Log-Odds (Logistic Regression)**:
   - *Severe Violation*: In a threshold classification setup (e.g., classifying points as "High Yield" vs "Low Yield"), optimal regions form localized "islands" or hyperspherical clusters surrounded by sub-optimal valleys. Standard logistic regression can only draw a single flat hyper-plane decision boundary, leading to severe underfitting unless augmented with high-degree polynomial or RBF kernels.
4. **Sample Size vs. Dimensionality (Sparsity)**:
   - In Function 8 (8D with only 41 samples) and Function 7 (6D with 31 samples), the feature-to-sample ratio is extremely low. Linear models fitted on sparse data suffer from high multicollinearity and parameter instability.

---

### Prompt 5: Are there any regions where the output appears roughly linear or where a decision boundary might form? How might a logistic regression classifier perform on this function, particularly in binary or threshold-based scenarios?

**Response**:
- **Roughly Linear Sub-Regions**:
  - While global linearity fails, our 1D sensitivity profile slices indicate that **localized sub-regions exhibit approximate monotonic linearity**. 
  - For example, in **Function 8 (8D)**, when fixing all dimensions at the current maximum and varying Dimension 1 or Dimension 3 within a narrow window ($\Delta x \in [0.0, 0.2]$), the output decreases linearly with coordinate values.
  - Similarly, in **Function 4 (4D)**, the local neighborhood surrounding $[0.38, 0.42, 0.40, 0.39]$ shows smooth, well-behaved local quadratic/linear slopes.
- **Performance of Logistic Regression in Threshold Scenarios**:
  - If we binarize the objective (e.g. $z = 1$ if $y > y_{threshold}$ and $0$ otherwise), logistic regression could successfully isolate the active search quadrant if the peak resides near one corner of the hypercube (as in Function 5 where $x_3 > 0.8, x_4 > 0.8 \implies z=1$).
  - However, in functions with concentric or multi-modal peaks (e.g. Function 3 and Function 7), a single linear decision boundary would misclassify sub-optimal regions on the opposite side of the hyper-plane, requiring non-linear kernel logistic regression or support vector machines.

---

### Prompt 6: Interpretability is a key advantage of linear and logistic regression. Did you find it useful to consider individual feature effects before deciding on your query point?

**Response**:
**Yes, analyzing individual feature effects was immensely useful and directly guided our query selection**:
1. **Feature Importance & Dimensional Screening**: Using Random Forest feature importance profiles (Panel 3 of our diagnostic plots), we identified that in **Function 5**, Dimensions 3 and 4 account for over $78\%$ of total output variance, whereas Dimensions 1 and 2 contribute less than $22\%$. This allowed us to prioritize optimizing $x_3$ and $x_4$ toward the high-yield upper boundary while maintaining moderate exploration on $x_1$ and $x_2$.
2. **1D Partial Dependence Slices**: Examining GP slice plots through the current maximum (Panel 2 of our diagnostic plots) revealed directional sensitivities. In **Function 8 (8D)**, the 1D slice demonstrated that Dimensions 1, 3, and 4 achieve maximum output when kept strictly close to zero ($x_i \approx 0.00-0.05$), while Dimensions 5 and 6 require higher coordinate values. Considering these individual feature behaviors eliminated over $90\%$ of unpromising candidate space and directly informed our Week 2 submission vector `0.000518-0.567012-0.060240-0.322979-0.466966-0.864631-0.061587-0.612956`.
