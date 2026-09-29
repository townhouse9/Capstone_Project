# Week 9 (Module 20) Reflection Report: Scaling, Emergence, and Robust Optimisation

## Required Capstone Component 20.1

### 1. Scaling Laws, Query Choices, and Marginal Returns
In black-box optimisation, empirical scaling laws govern how surrogate fidelity and decision quality evolve with sample size N and dimensionality d. Over nine sequential rounds, we observe two distinct regimes:

In low-dimensional spaces (Function 1 and Function 2 in 2D, N = 18), uniform spatial data scaling exhibits sharp diminishing returns. Once global topological contours are resolved, distributing additional points across the hypercube yields negligible information gain. On Function 2, progress has shifted entirely from exploratory scaling to targeted micro-exploitation around the local plateau (x = 0.698, y = 0.664), where our surrogate projects a refined peak of y = 0.754 +/- 0.091.

Conversely, in higher dimensions (Function 5 in 4D, Function 7 in 6D, and Function 8 in 8D), sample scaling yields steady, non-linear improvements when focused on active subspaces. Because hypercube volume scales exponentially with dimension, global coverage is impossible. However, scaling evaluations along sensitive coordinates identified via Automatic Relevance Determination (ARD) has unlocked dramatic performance gains. In Function 5, concentrating queries along the upper boundary ridge scaled outputs from 1088.86 initially to 2921.37, 5893.21, and 5430.88 this round. In Function 8, scaling along active coordinate axes yielded a 5-fold cross-validation R-squared of 0.990 and an all-time record peak of y = 9.951309.

### 2. Emergent Behaviours in Objective Landscapes
Emergent behaviours appear as sudden, discontinuous shifts in objective response that violate linear extrapolation. We observed this vividly in Function 4 (4D): after establishing a stable local maximum of y = +0.367529, exploratory steps in Weeks 8 and 9 plunged into a steep valley (y = -36.647771 and y = -1.479264). This revealed an emergent landscape topology: a narrow, isolated positive peak island surrounded by an unforgiving basin. Similarly, Function 5 revealed an emergent exponential ridge accelerating toward the upper vertex.

To prepare for these emergent phenomena, we deploy three structural defences:
1. Roughness-Tolerant Kernels: We prioritise Matérn 1.5 and 2.5 kernels with anisotropic ARD lengthscales over infinitely smooth RBF kernels, allowing the surrogate to model sharp non-linear phase changes without spurious oscillations.
2. Trust-Region Perturbation: We combine 70% space-filling Latin Hypercube candidates with 30% local Gaussian perturbations centred on historical optima, constraining step sizes near volatile cliffs.
3. Coordinate Clamping: All query vectors are strictly bounded within the unit hypercube (0.0 to 1.0), preventing extrapolative divergence at domain boundaries.

### 3. Trade-offs Between Cost, Robustness, and Performance
The physical evaluation budget—strictly one query per function per week—is the dominant economic and operational constraint. A single poor query wastes an entire iteration. Consequently, we deliberately trade speculative short-term gains for robust variance reduction:
1. Multi-Model Cross-Validation: Rather than relying on a single surrogate, we benchmark eight architectures (three ARD Gaussian Process variants, Random Forests, Extra Trees, Gradient Boosting, Polynomial Ridge, and Neural Network MLPs) across 5-fold cross-validation every round. This minor computational cost safeguards against model misspecification.
2. Distance De-duplication: We enforce a minimum Euclidean distance filter (min_dist = 0.005) against all historical evaluations, ensuring no query budget is spent re-evaluating known points.
3. Balanced Acquisition: We utilise Expected Improvement (EI) with range-scaled exploration jitter rather than greedy mean maximisation, ensuring queries balance peak pursuit against epistemic risk.

### 4. Balancing Predictable Optimisation with Uneven Emergence
Balancing steady, predictable progress against sudden emergent breakthroughs requires decoupling local exploitation from global curiosity. In functions where surrogate confidence is exceptionally high and landscapes are well-behaved (such as Function 8 with CV R-squared = 0.990, and Function 2 with CV R-squared = 0.445), we execute predictable, gradient-guided exploitation to climb established ridges.

In contrast, where landscapes exhibit emergent cliffs (Functions 3, 4, and 5), we temper aggressive extrapolation with uncertainty-penalised bounds. When Function 5's surrogate predicted an astronomical peak exceeding 6900 at the hypercube corner (1.0, 1.0, 1.0, 1.0), we verified that neighbouring evaluations (y = 5893.21 and y = 5430.88) physically substantiate an ascending ridge rather than an unconstrained hallucination. By tethering predictive modelling to rigorous cross-validation and maintaining bounded candidate perturbation, we capitalise on positive emergent leaps while insulating the campaign from catastrophic failure.
