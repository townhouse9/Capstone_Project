# Black-Box Optimization (BBO) - Week 6 (Module 17) Progress & Strategy Report
**Imperial College Capstone Project**

---

## 1. Week 6 Input Queries & Output Evaluation Summary

Following the evaluation of our Week 6 submissions, our optimization framework achieved its most successful round to date, securing **four brand-new all-time record global highs**:
- Function 5: Surged to an astronomical new peak of y = 4397.9532 (a massive +1476.58 gain in a single round over the previous 2921.37 peak).
- Function 7: Achieved a huge new record high of y = 2.233318 (up from 1.364968, a +0.868 gain).
- Function 6: Unlocked a new record high of y = -0.216645 (up from -0.305461).
- Function 3: Advanced to a new record high of y = -0.004807 (up from -0.012927).

### 1.1 Progression Table (Week 1 to Week 6)

| Function | Dim | Samples (N) | Previous Max y | W5 Evaluated y | Current Best y | Status / Gain | Week 6 Proposed Query Submission (x1-x2-...-xn) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| Func 1 | 2D | 15 | 7.71e-16 | 2.113e-53 | 7.71e-16 | Upper boundary mapped | 0.402010-0.055276 |
| Func 2 | 2D | 15 | 0.664342 | 0.556611 | 0.664342 | High plateau confirmed | 0.743719-0.095477 |
| Func 3 | 3D | 20 | -0.012927 | -0.004807 | -0.004807 | New Record High | 0.581486-0.424069-0.558637 |
| Func 4 | 4D | 35 | 0.367529 | -1.005934 | 0.367529 | Valley edge bounded | 0.316410-0.413201-0.373776-0.391902 |
| Func 5 | 4D | 25 | 2921.374749 | 4397.953210 | 4397.953210 | Astronomical Peak (+1476) | 0.969673-0.950156-0.973099-0.939200 |
| Func 6 | 5D | 25 | -0.305461 | -0.216645 | -0.216645 | New Record High | 0.465101-0.333351-0.824703-0.988817-0.216636 |
| Func 7 | 6D | 35 | 1.364968 | 2.233318 | 2.233318 | Huge Record High (+0.868) | 0.129335-0.330249-0.292549-0.378983-0.292339-0.784849 |
| Func 8 | 8D | 45 | 9.929387 | 9.898568 | 9.929387 | High ridge sustained | 0.080328-0.027267-0.110001-0.004206-0.326946-0.299329-0.110341-0.651118 |

---

## 2. Portal-Ready Reflection Questions and Answers (Module 17)

### Question 1
**Prompt**: CNNs build up features from edges and textures to full objects. How did this idea of progressive feature extraction influence the way you thought about refining your BBO strategy?

**Answer**:
In Convolutional Neural Networks, progressive feature extraction describes how receptive fields construct understanding hierarchically: initial layers detect low-level edges and color gradients, intermediate layers combine these into local geometric textures, and deeper layers compose textures into complete object representations.

This progressive abstraction directly mirrored how we refined our black-box optimization strategy across rounds:
1. Low-Level Coordinate Gradients (Edges): In initial iterations, we focused on isolated 1D coordinate sensitivity to detect which individual inputs created steep output gradients versus flat baselines.
2. Mid-Level Interaction Motifs (Textures): Next, we modeled pairwise non-linear interactions across active coordinate dimensions. In Function 5, we discovered that high output values depend on the joint placement of Dimension 2, Dimension 3, and Dimension 4 simultaneously above 0.90. In Function 8, we established that Dimensions 1, 3, and 7 must remain constrained near zero while searching along Dimensions 5 and 6.
3. High-Level Global Response Surface (Full Objects): Finally, our surrogate models (Gaussian Processes and Neural Networks) constructed a full multi-dimensional response manifold. Operating Expected Improvement acquisition over this holistic manifold enabled us to climb complex high-dimensional ridges, resulting in four record highs this round.

---

### Question 2
**Prompt**: LeNet and later CNNs redefined what is possible in computer vision. What parallels do you see between those breakthroughs and the incremental improvements you make in your BBO capstone project?

**Answer**:
Early vision breakthroughs like LeNet (1998) and later modern CNNs proved that combining structured inductive priors (such as local receptive fields, shared convolutional weights, and pooling translation invariance) with end-to-end backpropagation achieved performance leaps that unconstrained fully connected networks could never match.

We observed exact parallels in our capstone optimization journey:
1. Unstructured Sampling vs Incremental Progress: Random or unstructured search yields only slow, incremental improvements because it lacks spatial awareness of the underlying landscape.
2. Structural Priors vs Breakthrough Leaps: When we embedded structured mathematical priors into our pipeline—such as Matérn covariance kernel smoothness, dynamic range-scaled acquisition jitter, and neural network gradient backpropagation—our surrogate models achieved non-linear performance leaps. 

This structural refinement enabled dramatic surges across rounds: Function 5 escalated from an initial 1088.86 to 2062.99, then 2304.29, then 2921.37, and finally to an astronomical peak of y = 4397.95 this round. Similarly, Function 7 jumped from 1.364968 to a record y = 2.233318. Structured modeling transforms slow incremental exploration into high-yield breakthrough leaps.

---

### Question 3
**Prompt**: Training CNNs often involves balancing depth, computational costs and overfitting risks. Did you face similar trade-offs when choosing whether to explore widely or exploit promising regions in your queries?

**Answer**:
Yes, we faced direct parallels to the CNN architectural trade-off between network depth, computational cost, and overfitting when balancing exploration against exploitation in our weekly query selection:

1. Deep Exploitation (Analogous to Deep, High-Capacity CNNs):
Deeply exploiting a confirmed high-performing ridge provides rapid, high-magnitude output gains along established gradient paths. Concentrating our query budget on the upper ridge of Function 5 produced our massive 4397.95 peak, while exploiting Function 7 produced our 2.233318 record. However, just as overly deep CNNs risk overfitting to training noise and suffering from vanishing gradients, excessive exploitation risks getting permanently trapped in a sub-optimal local peak while ignoring unvisited global space.

2. Wide Exploration (Analogous to Broad Receptive Fields and Regularization):
Broad exploration samples unvisited hypercube quadrants with high posterior variance, ensuring global coverage and guarding against missed global optima. However, like wide shallow networks with high computational overhead, exploration consumes expensive weekly queries without immediately maximizing the top output score.

We balanced this trade-off using cross-validation surrogate confidence: deeply exploiting high-confidence ridges on surging functions (Functions 5 and 7), while maintaining exploratory sampling on uncertain functions (Functions 1 and 4).

---

### Question 4
**Prompt**: Convolutions, pooling, activations and loss functions influence how CNNs learn from data. Which of these concepts helped you think differently about how your optimisation model learns from your accumulated data?

**Answer**:
Four core CNN building blocks reshaped how we conceptualize surrogate model learning:

1. Convolutions (Local Spatial Receptive Fields):
Convolutions taught us that data points exert strong localized influence. This reinforced our use of Matérn Gaussian Process kernels, where covariance between observations decays smoothly with Euclidean distance, ensuring local peak structure is preserved without distortion from distant points.

2. Pooling (Spatial Downsampling and Invariance):
Pooling inspired candidate space binning during Monte Carlo acquisition. By downsampling low-probability regions and concentrating candidate density around high-variance boundaries, we reduced candidate evaluation cost while maintaining high spatial resolution near peaks.

3. Non-Linear Activations (Threshold Mechanics):
Activation functions like ReLU demonstrated that response surfaces undergo sharp phase transitions when coordinate thresholds are crossed. In Function 5 and Function 7, outputs remain modest until coordinates cross critical boundaries (such as Dimension 3 and Dimension 4 exceeding 0.90), triggering exponential growth.

4. Loss Functions (Multi-Scale Variance Regularization):
Evaluating surrogate prediction error across functions spanning scales from 1e-53 (Function 1) to 4397.95 (Function 5) highlighted the need for target normalization and range-scaled loss penalties to prevent high-magnitude functions from dominating model training.

---

### Question 5
**Prompt**: The interview with Andrea Dunbar highlighted the trade-offs of deploying CNNs in edge AI systems. How might reflecting on real-world deployment challenges help you decide how to benchmark success in your own BBO capstone project?

**Answer**:
In Andrea Dunbar's discussion of Edge AI deployment, embedding CNNs on low-power, memory-constrained edge hardware requires managing strict engineering trade-offs between inference latency, energy consumption, memory footprint, and classification accuracy. A massive model with high theoretical accuracy is useless if it exceeds the latency budget or drains the system battery.

Reflecting on real-world Edge AI deployment informed how we benchmark success in our capstone project:
1. Beyond Offline Validation Metrics: Having a surrogate model with a low offline Cross-Validation Mean Squared Error is merely a secondary diagnostic. In BBO, a model is only successful if its acquisition recommendations reliably direct queries to higher observed outputs under strict evaluation limits.
2. Decision Efficiency Under Hard Constraints: In Edge AI, memory and power are hard constraints. In our capstone, the evaluation budget (one sample per function per week) is our hard constraint. Success is benchmarked by decision efficiency: how effectively our surrogate engine translates sparse historical data into record-breaking physical outputs.

Benchmarking success against decision efficiency under strict constraints aligns our capstone process directly with real-world industrial AI deployment standards.
