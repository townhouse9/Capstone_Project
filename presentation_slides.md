# BBO Capstone Presentation: Methodology, Evolution, and Insights
**Imperial College London — Black-Box Optimization (BBO) Capstone Project**

---

### Slide 1: An Overview of Your BBO Approach

#### Core Objective and Purpose
The fundamental objective of this project is to discover the global maximum of eight unknown, computationally expensive black-box mathematical functions operating within continuous unit hypercubes spanning two to eight dimensions. Because we are denied analytical formulas, functional gradients, and internal landscape visibility, and are restricted to an exacting budget of exactly one evaluation per function per weekly round, our goal is to achieve maximal sample efficiency by discovering optimal input coordinates with minimal query expenditure.

#### Overall Strategy and Narrative Architecture
Our optimisation framework operates as an automated, closed-loop Bayesian active learning pipeline that constructs probabilistic surrogate models of the hidden response landscapes from empirical evaluation histories. At each iteration, the pipeline executes rigorous five-fold cross-validation across a diverse model suite—incorporating Automatic Relevance Determination Gaussian Processes, multi-layer perceptron neural networks, and tree-based ensembles—to identify the most reliable surrogate representation. Candidate queries are generated through a hybrid mechanism combining quasi-random Latin Hypercube exploration with localized perturbation envelopes around historical elite points, filtered through strict Euclidean distance de-duplication to prevent wasted queries, and selected via Expected Improvement and Upper Confidence Bound acquisition criteria.

---

### Slide 2: How Your Strategy Has Evolved

#### Key Evolution Since Early Rounds
Our strategy has transitioned from broad, uninformed quasi-random exploration into an anisotropic, variance-driven active learning system characterised by structured mathematical discipline. In early iterations, when sparse observations precluded dependable surrogate fitting, our process relied on wide Latin Hypercube sweeps governed by uniform spatial priors and high epistemic uncertainty across all dimensions. As empirical observations expanded past twenty points per function, we abandoned isotropic assumptions in favour of coordinate-specific sensitivity modeling, replacing unguided perturbations with structured active subspace exploitation.

#### Influencing Drivers and Feedback Mechanisms
This methodological evolution was driven by empirical performance trends, rigorous cross-validation diagnostics, and the persistent curse of dimensionality observed in moderate-to-high dimensions. Treating all coordinate axes identically diluted our sparse query budget across uninformative directions and caused models to stagnate against boundary walls. Cross-validation predictive metrics revealed that well-calibrated Gaussian Processes with anisotropic kernels consistently outperformed unregularised deep regressors in low-data regimes, directly motivating our decision to anchor decisions to probabilistic lengthscale telemetry.

#### Guiding Principles and Heuristics
Our current query selection is governed by four core heuristics:
1. Automatic Relevance Determination to isolate sensitive active subspaces from invariant dimensions.
2. Mandatory Euclidean distance de-duplication ensuring a minimum separation of 0.005 from all historical evaluations to preserve novel informational gain.
3. Adaptive acquisition fallbacks that transition from Expected Improvement to Upper Confidence Bound whenever improvement gradients diminish near saturated regions.
4. Multi-model predictive calibration, prioritizing surrogates with superior cross-validation generalisation over raw training accuracy.

---

### Slide 3: Patterns, Data and Insights

#### Meaningful Empirical Trends Across Data Points
Analysing our cumulative dataset across 263 evaluations revealed that objective response landscapes are rarely uniform; rather, they partition into low-dimensional active manifolds, steep localized ridges, and narrow ascending basins nestled within broader penalty plateaus. This structural insight underpinned our breakthrough twelfth round, where focused exploitation unlocked three brand-new all-time record highs: Function 6 ascending to -0.202865, Function 7 surging to an unprecedented 2.661908, and Function 8 achieving an elite peak of 9.966558.

#### Key Influencing Variables and Behaviours
Performance variation is overwhelmingly dictated by low-dimensional principal coordinate combinations rather than full nominal dimensional spaces. In Function 8, spanning eight dimensions, the primary driver of variation is an active subspace where the first two coordinates collapse towards zero while the fifth coordinate nears unity, governing the entire path towards our 9.966558 maximum. Similarly, Function 5 displays an extreme directional gradient that drives objective values upwards by orders of magnitude (from 171 to 8662) as coordinates simultaneously approach the upper vertex (1.0, 1.0, 1.0, 1.0), whereas Function 4 features a delicate positive centroid island (centred near 0.40, 0.42, 0.38, 0.34) surrounded by severe negative penalties.

#### Topographical Understanding of the Search Process
These empirical observations have profoundly reshaped our mental model of black-box optimisation, proving that apparent high-dimensional complexity frequently conceals low-dimensional intrinsic structures. By treating input spaces through the lens of coordinate relevance and identifying directional gradients, we transformed our search from blind spatial sampling into a principled ascent along dominant topological corridors.

---

### Slide 4: Decision-Making and Iteration

#### Balancing Exploration and Exploitation
We govern the balance between exploration and exploitation through probabilistic acquisition functions that mathematically trade off predictive mean (exploitation) against epistemic posterior variance (exploration). In earlier iterations, broad uncertainty envelopes guided queries towards undersampled regions to uncover landscape contours; however, as rounds advanced and high-yield basins were verified, we systematically tightened candidate perturbation radii around elite clusters, using range-scaled Expected Improvement to concentrate search density where expected returns were highest.

#### Strategic Decisions: Successes and Shortcomings
- What Worked: Isolating the active subspace in Function 8 and concentrating queries within the ascending basin of Function 7 produced consecutive record-breaking peaks (2.661908 and 9.966558), while targeting the extreme upper vertex in Function 5 unlocked our monumental 8662.482500 score.
- What Did Not Work: Early isotropic perturbations in high-dimensional spaces frequently dropped evaluations into catastrophic penalty valleys (such as Function 4 collapsing below -36), and unscaled Expected Improvement occasionally caused candidate clustering along domain boundaries without probing interior maxima.

#### Handling Uncertainty and Adapting to Negative Surprises
When an evaluated query yields an unexpected drop or contradicts surrogate predictions, our framework treats the outcome as critical epistemic evidence that refines our posterior belief. Rather than overreacting with chaotic parameter changes, the system incorporates the negative point to update covariance lengthscales, inflates predictive uncertainty across the surrounding neighbourhood to deter immediate re-sampling, and activates distance-constrained orthogonal candidates to explore alternate topographical hypotheses.

---

### Slide 5: Next Steps and Reflection

#### Planned Actions for the Final Round (Module 24)
With Module 24 representing the terminal query round of the entire capstone campaign, the future value of acquiring exploratory information drops to zero, rendering speculative variance-seeking irrational under finite-horizon decision theory. Consequently, our final submission strategy transitions decisively to pure exploitation: we will compress candidate perturbation envelopes tightly around our confirmed global peaks (specifically Functions 6, 7, and 8), reduce exploratory acquisition weights, and employ fine-grained numerical gradient climbing along dominant principal axes to extract the final increments of objective performance.

#### Broader Machine Learning Context
This sequential optimisation challenge directly mirrors fundamental real-world machine learning problems, including automated hyperparameter tuning for multi-billion-parameter foundation models, chemical molecular discovery, and physical robotic control systems where live trials are exceptionally hazardous or expensive. In all these critical applications, the core principles developed here—rigorous uncertainty quantification, sample-efficient active learning, and inductive bias selection—form the bedrock of modern artificial intelligence engineering.

#### Communicating Results to Non-Technical Stakeholders
"Imagine searching for the highest mountain peak in an unfamiliar landscape shrouded in dense fog, where every single step takes a full week and costs substantial resources; rather than wandering blindly through the mist, our intelligent framework uses mathematical radar to build an increasingly accurate virtual map from every footstep, systematically navigating through the terrain to guide us straight to the highest summit with maximum efficiency and minimum risk."
