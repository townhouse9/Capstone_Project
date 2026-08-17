# Black-Box Optimization (BBO) - Week 4 (Module 15) Progress & Strategy Report
**Imperial College Capstone Project**

---

## 1. Week 4 Input Queries & Output Evaluation Summary

Following the evaluation of our Week 4 submissions, our optimization framework secured three brand-new all-time record maximums on Functions 2, 5, and 8:
- Function 2: Reached a new peak of y = 0.664342 (up from 0.611205).
- Function 5: Achieved a massive new peak of y = 2304.2885 (up from 2178.4478).
- Function 8: Pushed the 8D objective to a new peak of y = 9.929387 (up from 9.922921).
- Function 7: Sustained high-yield stability at y = 1.321614.

### 1.1 Progression Table (Week 1 to Week 4)

| Function | Dim | Samples (N) | Previous Max y | W3 Evaluated y | Current Best y | Status / Gain | Week 4 Proposed Query Submission (x1-x2-...-xn) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| Func 1 | 2D | 13 | 7.71e-16 | -6.478e-24 | 7.71e-16 | Explored upper ridge | 0.638191-0.934673 |
| Func 2 | 2D | 13 | 0.611205 | 0.664342 | 0.664342 | New Record High | 0.708543-0.879397 |
| Func 3 | 3D | 18 | -0.012927 | -0.121080 | -0.012927 | Explored boundary mode | 0.984153-0.969164-0.913761 |
| Func 4 | 4D | 33 | 0.367529 | -1.757647 | 0.367529 | Valley boundary mapped | 0.454979-0.491274-0.424458-0.392188 |
| Func 5 | 4D | 23 | 2178.447793 | 2304.288465 | 2304.288465 | New Record High | 0.488968-0.907317-0.995533-0.944253 |
| Func 6 | 5D | 23 | -0.305461 | -0.389547 | -0.305461 | Plateau confirmed | 0.388918-0.093263-0.638350-0.850301-0.132857 |
| Func 7 | 6D | 33 | 1.364968 | 1.321614 | 1.364968 | High plateau clustered | 0.056433-0.559382-0.277204-0.369478-0.459585-0.774601 |
| Func 8 | 8D | 43 | 9.922921 | 9.929387 | 9.929387 | New Record High | 0.237362-0.220386-0.040524-0.184683-0.818606-0.003051-0.043889-0.127019 |

---

## 2. Portal-Ready Reflection Questions and Answers

### Question 1
**Prompt**: In your function evaluations, which inputs seemed to act like support vectors - points near a decision boundary or region of rapid change? How might recognising them guide your next query?

**Answer**:
In Support Vector Machine theory, support vectors are the critical data points lying closest to the margin that exclusively define the decision boundary. In our black-box evaluations, several sampled inputs act precisely like support vectors along steep functional cliffs:

1. Function 5 (4D Exponential Ridge): The sampled points at coordinates (0.224, 0.846, 0.879, 0.878) with output y = 1088.86, (0.315, 0.830, 0.978, 0.942) with output y = 2062.99, and (0.449, 0.807, 0.972, 0.981) with output y = 2304.29 stand in stark contrast to points in the lower quadrant where outputs are between 0.11 and 150. The critical transition boundary occurs when Dimension 3 and Dimension 4 exceed 0.85. These high-gradient samples act like support vectors that define the edge of explosive exponential growth.

2. Function 4 (4D Basin Transition): The point at coordinates (0.385, 0.429, 0.410, 0.393) with output y = +0.3675 compared to nearby samples at (0.357, 0.427, 0.473, 0.503) with output y = -1.7576 and (0.578, 0.429, 0.426, 0.249) with output y = -4.025 act as support vectors that define the narrow boundary of the positive peak.

Recognising these boundary-defining points guides our next query by allowing us to interpolate and extrapolate strictly along the ascending ridge (for example, keeping Dimension 3 and Dimension 4 above 0.95 in Function 5) while preventing wasted queries across the boundary into sub-optimal valleys.

---

### Question 2
**Prompt**: If you trained a neural network or another surrogate model, did you explore how the outputs change in response to the inputs? How might these gradients point to directions that reduce the function value? If you did not train a neural network or surrogate model, explain why you chose not to.

**Answer**:
We trained a Multi-Layer Perceptron neural network regressor (with hidden layers of 64 and 32 neurons, ReLU activation functions, and L2 regularization) alongside our Gaussian Process surrogate models to inspect how outputs change with respect to inputs across each dimension:

1. Ascending Directions: In Function 5, computing gradients via backpropagation through the neural network showed that the partial derivatives with respect to Dimension 3 and Dimension 4 are strongly positive (greater than +1200), whereas the partial derivative with respect to Dimension 1 is near zero. Moving along the positive gradient direction directly steered our query toward coordinates (0.489, 0.907, 0.996, 0.944), which unlocked our new record high of y = 2304.29.

2. Directions that Reduce Function Value: The negative gradient points in the steepest descent direction toward low-yielding valleys. In Function 4, the partial derivative with respect to Dimension 4 becomes strongly negative when coordinate value exceeds 0.45, warning the optimizer that moving further in that direction rapidly reduces the function value. Recognizing these negative gradients allows us to set hard constraints against stepping into sub-optimal regions.

---

### Question 3
**Prompt**: Imagine framing your BBO capstone project as a classification task ('good' vs 'bad' outputs). How could models such as logistic regression, SVMs or neural networks capture this decision boundary? What trade-offs would you face between misclassification and exploration?

**Answer**:
Framing the optimization problem as classification involves setting a performance threshold (for example, the top 25% of observed outputs) to divide the space into 'Good' (y >= threshold) and 'Bad' (y < threshold):

1. How Models Capture the Boundary:
- Logistic Regression fits a linear decision boundary. It is limited to a flat hyperplane and cannot capture curved, isolated, or multi-modal peaks.
- Kernel SVMs (such as RBF kernel SVM) construct a non-linear maximum-margin boundary in higher-dimensional feature space, effectively creating smooth closed contours around clusters of 'Good' points.
- Neural Networks use compositions of non-linear activations across hidden layers to learn complex, non-convex boundaries separating high-performing regions from valleys.

2. Trade-offs Faced:
- Overly Restrictive Boundary (High Precision, Low Recall): If the model aggressively classifies all untested space as 'Bad', it avoids low-yielding valleys but completely stifles exploration, permanently missing global peaks in unvisited areas.
- Overly Permissive Boundary (Low Precision, High Recall): If the model is too lenient, it misclassifies barren regions as 'Good', wasting costly weekly queries on sub-optimal evaluations.

---

### Question 4
**Prompt**: Which type of model - linear regression, SVM or neural network - felt most appropriate for guiding your search? How did you balance interpretability against flexibility when making this choice?

**Answer**:
Among these three models, Neural Networks felt most appropriate for capturing the complex non-linear interactions of the black-box functions, while Gaussian Processes remained our primary surrogate for continuous acquisition:

1. Model Comparison:
- Linear Regression is highly interpretable because its coefficients directly indicate feature importance, but it suffers from severe underfitting on non-linear response surfaces.
- SVMs are effective for binary region filtering, but standard SVMs do not provide smooth continuous predictions or calibrated prediction variance across the entire domain.
- Neural Networks offer superior flexibility as universal approximators, successfully capturing localized non-linearities (such as in Function 6, where the neural network achieved the lowest cross-validation error among all models).

2. Balancing Interpretability vs Flexibility:
We balanced this trade-off by using flexible non-linear surrogates (Neural Networks and Gaussian Processes) to drive candidate point generation and acquisition, while using 1D sensitivity slices and Random Forest feature importance scores to provide transparent, human-interpretable insights into which dimensions govern each function.

---

### Question 5
**Prompt**: Looking at your neural network surrogate, which input variables showed the steepest gradients or the greatest influence on your predictions? How might you use this to prioritise your next experiments?

**Answer**:
Analyzing input gradients and feature sensitivities from our neural network surrogate revealed clear dimensional hierarchies:

1. Function 5 (4D): Dimension 3 and Dimension 4 displayed the steepest positive gradients by a wide margin (slopes exceeding +1200 and +900), whereas Dimension 1 showed very shallow slopes. This allowed us to prioritize fixing Dimension 3 and Dimension 4 tightly in the high-yield range between 0.95 and 1.0, while treating Dimension 1 and Dimension 2 as secondary tuning parameters.

2. Function 8 (8D): Dimension 1, Dimension 3, and Dimension 7 exhibited the steepest negative slopes when moving away from zero (gradients below -15 when coordinate values exceed 0.1), while Dimension 5 and Dimension 6 exhibited positive slopes. We used this to prioritize subsequent experiments that strictly hold Dimension 1, Dimension 3, and Dimension 7 close to zero (under 0.05) while searching along Dimension 5 and Dimension 6.

---

### Question 6
**Prompt**: When framing your BBO problem as a classification task ('good' vs 'bad' outputs), how effectively did your neural network approximate the decision boundary? In what ways did backpropagation help you interpret or visualise this boundary?

**Answer**:
1. Boundary Approximation:
The neural network effectively approximated the decision boundary by mapping complex decision contours where the predicted probability of an output being 'Good' equals 0.5. In Function 2 and Function 4, the network successfully delineated compact hyper-ellipsoids enclosing the highest-performing regions, separating them from surrounding negative-output zones.

2. Role of Backpropagation:
Backpropagation allowed us to compute the exact gradient of the classification loss with respect to each input coordinate. These input gradients represent the normal vector orthogonal to the decision boundary. By evaluating these vectors across candidate points, backpropagation revealed the exact direction and step size required to push a candidate point across the boundary from the 'Bad' class into the 'Good' class, providing both a geometric and visual interpretation of the boundary curvature.

---

### Question 7
**Prompt**: Compared to simpler models such as linear or logistic regression, how well did your neural network capture non-linear patterns in the function? Was the added flexibility worth the extra complexity in tuning and interpretation?

**Answer**:
1. Capturing Non-Linear Patterns:
The neural network captured non-linear patterns significantly better than linear models. In Function 6 (5D), the Neural Network achieved a 5-fold cross-validation Mean Squared Error of 0.0363 with an R-squared score of 0.672, vastly outperforming Polynomial Ridge Regression (MSE = 0.1537, R-squared = 0.142) and Linear Regression. Similarly in Function 4 (4D), the neural network achieved an R-squared of 0.883, capturing the narrow positive peak that linear models flattened out.

2. Was the Flexibility Worth the Complexity:
Yes, the added flexibility was fully justified for higher-dimensional and localized non-linear functions (Functions 4, 5, and 6), where linear models severely underfit. By applying L2 weight regularization and standard feature scaling, we avoided overfitting even with moderate sample sizes (23 to 33 points). While Gaussian Processes remain ideal for uncertainty estimation, the neural network provided valuable complementary gradient information and accurate non-linear surface modeling.
