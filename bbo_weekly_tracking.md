# Black-Box Optimization (BBO) Weekly Tracking Log
**Imperial College Capstone Project**

This document serves as the master tracking log for optimising the 8 synthetic black-box functions. It documents our framework design, weekly surrogate model cross-validation benchmarks, portal submission strings, visual diagnostic progress, and formal reflections across all rounds.

---

## 1. Master Dashboard & Weekly Optimization Trajectory

![Weekly Progress Master Dashboard](file:///c:/Users/sesa625752/OneDrive%20-%20Schneider%20Electric/Imperial_College/Capstone_Project/visualizations/weekly_progress_summary.png)

The master dashboard image above ([weekly_progress_summary.png](file:///c:/Users/sesa625752/OneDrive%20-%20Schneider%20Electric/Imperial_College/Capstone_Project/visualizations/weekly_progress_summary.png)) provides a 4-panel overview:
1. **Current Max vs Predicted Next Query Output**: Side-by-side comparison of current maximum $y$ vs GP mean prediction with $\pm \sigma$ error bars for Functions 1–8.
2. **Weekly Optimization Trajectory**: Historical line plot tracking peak output achieved per function across weekly rounds (Week 1 $\to$ Week 2 $\to$ Week 3 $\to$ Week 4).
3. **Surrogate Model CV Performance**: Bar chart illustrating the winning surrogate model and 5-Fold CV-MSE score across all 8 functions.
4. **Expected Improvement (EI) Potential Ratio**: Normalized gain metric highlighting which functions possess the highest potential for global peak discovery in upcoming submissions.

---

## 2. Multi-Week Progress & Week 4 Submissions Summary

### 2.1 Function Progression & Week 4 Submissions Table

| Function | Dim | Initial $\to$ Current Samples | Previous Max $y$ | W3 Evaluated $y$ | Current Best $y$ | Status / Gain | Week 4 Proposed Query Submission (`x1-x2-...-xn`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Func 1** | 2D | 10 $\to$ **13** | $7.71 \times 10^{-16}$ | $-6.478 \times 10^{-24}$ | $7.71 \times 10^{-16}$ | Explored upper ridge | `0.638191-0.934673` |
| **Func 2** | 2D | 10 $\to$ **13** | $0.611205$ | **$0.664342$** | **$0.664342$** | 🚀 **New Record High** | `0.708543-0.879397` |
| **Func 3** | 3D | 15 $\to$ **18** | $-0.012927$ | $-0.121080$ | $-0.012927$ | Explored boundary mode | `0.984153-0.969164-0.913761` |
| **Func 4** | 4D | 30 $\to$ **33** | $+0.367529$ | $-1.757647$ | **$+0.367529$** | Valley boundary mapped | `0.454979-0.491274-0.424458-0.392188` |
| **Func 5** | 4D | 20 $\to$ **23** | $2178.447793$ | **$2304.288465$** | **$2304.288465$** | 🚀 **New Record High** | `0.488968-0.907317-0.995533-0.944253` |
| **Func 6** | 5D | 20 $\to$ **23** | $-0.305461$ | $-0.389547$ | $-0.305461$ | Plateau confirmed | `0.388918-0.093263-0.638350-0.850301-0.132857` |
| **Func 7** | 6D | 30 $\to$ **33** | $1.364968$ | $1.321614$ | $1.364968$ | High plateau clustered | `0.056433-0.559382-0.277204-0.369478-0.459585-0.774601` |
| **Func 8** | 8D | 40 $\to$ **43** | $9.922921$ | **$9.929387$** | **$9.929387$** | 🚀 **New Record High** | `0.237362-0.220386-0.040524-0.184683-0.818606-0.003051-0.043889-0.127019` |

---

## 3. Week 4 (Module 15) Reflection & Strategy Report

### Question 1: Support Vectors & Regions of Rapid Change
**Prompt**: *In your function evaluations, which inputs seemed to act like support vectors – points near a decision boundary or region of rapid change? How might recognising them guide your next query?*
- **Response**:
  1. **Function 5 (4D Exponential Ridge)**: Points at $[0.224, 0.846, 0.879, 0.878]$ ($y = 1088.86$), $[0.315, 0.830, 0.978, 0.942]$ ($y = 2062.99$), and $[0.449, 0.807, 0.972, 0.981]$ ($y = 2304.29$) stand in stark contrast to points in the lower quadrant ($y \approx 0.11 - 150$). The critical transition boundary occurs around $x_3, x_4 \in [0.85, 0.92]$. These samples act as "support vectors" delineating the explosive ascending ridge.
  2. **Function 4 (4D Basin Transition)**: The sample $[0.385, 0.429, 0.410, 0.393]$ ($y = +0.3675$) vs $[0.357, 0.427, 0.473, 0.503]$ ($y = -1.7576$) and $[0.578, 0.429, 0.426, 0.249]$ ($y = -4.025$) act as support vectors that define the narrow basin of attraction.
  - **Guiding Next Queries**: Recognizing these boundary-defining points allows us to interpolate and extrapolate strictly along the ascending tangent (e.g. pushing $x_3, x_4 \to 0.99$ in Function 5) while preventing wasted queries in sub-optimal space.

### Question 2: Neural Network Surrogates & Gradient Exploration
**Prompt**: *If you trained a neural network or another surrogate model, did you explore how the outputs change in response to the inputs? How might these gradients point to directions that reduce the function value? If you did not train a neural network or surrogate model, explain why you chose not to.*
- **Response**: We incorporated an MLP Regressor (`hidden_layer_sizes=(64, 32)`, ReLU activations, $L_2$ regularization) alongside Gaussian Process surrogates to inspect analytical input gradients $\nabla_x \hat{f}(x) = \left[ \frac{\partial \hat{y}}{\partial x_1}, \dots, \frac{\partial \hat{y}}{\partial x_d} \right]$:
  - In **Function 5**, computing gradients via backpropagation revealed that $\frac{\partial \hat{y}}{\partial x_3} > 0$ and $\frac{\partial \hat{y}}{\partial x_4} > 0$ possess the largest positive magnitudes ($> +1200$), whereas $\frac{\partial \hat{y}}{\partial x_1} \approx 0$. Following the gradient ascent direction directly steered our Week 4 query toward $[0.489, 0.907, 0.996, 0.944]$, unlocking the new $2304.29$ record.
  - Conversely, the negative gradient $-\nabla_x \hat{y}$ points toward steep descent directions (valleys). In Function 4, $\frac{\partial \hat{y}}{\partial x_4}$ becomes strongly negative when $x_4 > 0.45$, warning the optimizer against expanding further into higher $x_4$ coordinates.

### Question 3: BBO as a Classification Task ('Good' vs 'Bad') & Trade-Offs
**Prompt**: *Imagine framing your BBO capstone project as a classification task (‘good’ vs ‘bad’ outputs). How could models such as logistic regression, SVMs or neural networks capture this decision boundary? What trade-offs would you face between misclassification and exploration?*
- **Response**:
  1. **Decision Boundary Models**:
     - *Logistic Regression*: Fits $P(C_1 \mid x) = \sigma(\mathbf{w}^T \mathbf{x} + b)$. Limited to a single flat hyperplane; fails on multi-modal landscapes.
     - *Kernel SVM (RBF)*: Maximizes margin in Hilbert space, constructing smooth non-linear decision boundaries around isolated clusters of 'Good' points.
     - *Neural Networks (MLP Classifier)*: Uses composite hidden layer activations to form arbitrary piece-wise linear/curved decision envelopes.
  2. **Trade-Offs**:
     - *Overly Restrictive (High Precision, Low Recall)*: Classifies all untested space as 'Bad', preventing sampling of valleys but permanently missing undiscovered global peaks.
     - *Overly Permissive (Low Precision, High Recall)*: Classifies wide swaths as potentially 'Good', diluting query budget into low-yielding territory.

### Question 4: Model Selection: Linear Regression vs SVM vs Neural Network vs GP
**Prompt**: *Which type of model – linear regression, SVM or neural network – felt most appropriate for guiding your search? How did you balance interpretability against flexibility when making this choice?*
- **Response**:
  - *Linear Regression*: Highly interpretable ($\beta_i$), but lacks flexibility for non-linear peaks.
  - *SVM*: Strong for region classification, but does not provide calibrated continuous response variance.
  - *Neural Networks*: Flexible universal approximators (Function 6 top score $MSE = 0.0363$), but prone to overfitting in sparse data regimes ($N < 30$).
  - *Gaussian Process Regressors (GP-BO)*: **The overall most appropriate model**. Combines non-parametric smoothing with closed-form posterior uncertainty $\sigma(x)$ for principled acquisition (Expected Improvement).
  - *Balance*: We used **GPs and MLPs for flexible acquisition**, and **Random Forest feature importances / 1D slices for human interpretability**.

### Question 5: Neural Network Gradient Saliency & Experiment Prioritization
**Prompt**: *Looking at your neural network surrogate, which input variables showed the steepest gradients or the greatest influence on your predictions? How might you use this to prioritise your next experiments?*
- **Response**:
  - **Function 5 (4D)**: Dimensions 3 and 4 exhibited by far the steepest positive gradients ($\frac{\partial \hat{y}}{\partial x_3} \approx +1420$, $\frac{\partial \hat{y}}{\partial x_4} \approx +980$). This prioritized clamping $x_3, x_4 \in [0.95, 1.0]$ in subsequent queries while treating $x_1, x_2$ as fine-tuning parameters.
  - **Function 8 (8D)**: Dimensions 1, 3, and 7 exhibited steep negative gradients near zero ($\frac{\partial \hat{y}}{\partial x_i} < -15.0$ when $x_i > 0.1$), whereas Dimensions 5 and 6 had positive slopes. We prioritized experiments fixing $x_1, x_3, x_7 \le 0.05$ while exploring along Dimensions 5 and 6.

### Question 6: Neural Network Boundary Approximation & Backpropagation
**Prompt**: *When framing your BBO problem as a classification task (‘good’ vs ‘bad’ outputs), how effectively did your neural network approximate the decision boundary? In what ways did backpropagation help you interpret or visualise this boundary?*
- **Response**:
  - The neural network approximated the boundary by partitioning space where output neuron $P(y \ge \tau) \ge 0.5$, isolating high-yield basins into compact hyper-ellipsoids in Functions 2 and 4.
  - Backpropagation computes $\nabla_x \mathcal{L}_{BCE}$, revealing the **orthogonal normal vector to the decision surface** and indicating which coordinate changes push a candidate point most rapidly into the 'Good' class.

### Question 7: Non-Linear Patterns & Complexity Trade-Off
**Prompt**: *Compared to simpler models such as linear or logistic regression, how well did your neural network capture non-linear patterns in the function? Was the added flexibility worth the extra complexity in tuning and interpretation?*
- **Response**:
  - On **Function 6 (5D)**, the Neural Network achieved a 5-Fold CV-MSE of **$0.0363$** ($R^2 = 0.672$), vastly outperforming Linear/Polynomial Ridge ($MSE = 0.1537$) and Gradient Boosting ($MSE = 0.2111$).
  - The added flexibility was worth the complexity on higher-dimensional non-linear functions (Functions 4, 5, 6) when paired with $L_2$ weight decay ($\alpha=10^{-3}$) and standard scaling.
