# Black-Box Optimization (BBO) Weekly Tracking Log
**Imperial College Capstone Project**

This document serves as the master tracking log for optimising the 8 synthetic black-box functions. It documents our framework design, weekly surrogate model cross-validation benchmarks, portal submission strings, visual diagnostic progress, and formal reflections across all rounds.

---

## 1. Master Dashboard & Weekly Optimization Trajectory

![Weekly Progress Master Dashboard](file:///c:/Users/sesa625752/OneDrive%20-%20Schneider%20Electric/Imperial_College/Capstone_Project/visualizations/weekly_progress_summary.png)

The master dashboard image above ([weekly_progress_summary.png](file:///c:/Users/sesa625752/OneDrive%20-%20Schneider%20Electric/Imperial_College/Capstone_Project/visualizations/weekly_progress_summary.png)) provides a 4-panel overview:
1. Current Max vs Predicted Next Query Output: Side-by-side comparison of current maximum y vs GP mean prediction with error bars for Functions 1 to 8.
2. Weekly Optimization Trajectory: Historical line plot tracking peak output achieved per function across weekly rounds (Week 1 to Week 5).
3. Surrogate Model CV Performance: Bar chart illustrating the winning surrogate model and 5-Fold CV-MSE score across all 8 functions.
4. Expected Improvement (EI) Potential Ratio: Normalized gain metric highlighting which functions possess the highest potential for global peak discovery in upcoming submissions.

---

## 2. Multi-Week Progress & Week 5 Submissions Summary

### 2.1 Function Progression & Week 5 Submissions Table

| Function | Dim | Initial to Current Samples | Previous Max y | W4 Evaluated y | Current Best y | Status / Gain | Week 5 Proposed Query Submission (x1-x2-...-xn) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| Func 1 | 2D | 10 to 14 | 7.71e-16 | 1.145e-65 | 7.71e-16 | Upper boundary mapped | 0.321608-0.165829 |
| Func 2 | 2D | 10 to 14 | 0.664342 | 0.656877 | 0.664342 | High plateau confirmed | 0.678392-0.879397 |
| Func 3 | 3D | 15 to 19 | -0.012927 | -0.125720 | -0.012927 | Explored corner mode | 0.455167-0.521242-0.486792 |
| Func 4 | 4D | 30 to 34 | 0.367529 | -1.514948 | 0.367529 | Valley edge bounded | 0.437086-0.290257-0.382594-0.349796 |
| Func 5 | 4D | 20 to 24 | 2304.288465 | 2921.374749 | 2921.374749 | Massive Record High (+617) | 0.595561-0.993735-0.987345-0.981682 |
| Func 6 | 5D | 20 to 24 | -0.305461 | -0.448204 | -0.305461 | Sub-space mapped | 0.437161-0.320742-0.578847-0.762968-0.194246 |
| Func 7 | 6D | 30 to 34 | 1.364968 | 0.899910 | 1.364968 | Clustered peak zone | 0.105640-0.347690-0.307992-0.359223-0.329529-0.769180 |
| Func 8 | 8D | 40 to 44 | 9.929387 | 9.613845 | 9.929387 | High ridge sustained | 0.050323-0.062907-0.187347-0.032471-0.743353-0.723330-0.136056-0.836079 |

---

## 3. Week 5 (Module 16) Reflection & Strategy Report (Portal-Ready)

### Question 1: Hierarchical Feature Learning in Optimization
In deep learning, hierarchical feature learning describes how neural networks construct understanding in stages: early layers identify basic low-level features, middle layers assemble these into motifs, and deeper layers combine motifs into high-level holistic concepts.

This directly influenced our multi-tiered optimization hierarchy:
1. Low-Level Dimensional Screening: First, we analyzed coordinate sensitivity to separate active driving dimensions from inactive parameters. For example, in Function 5, Dimensions 3 and 4 dominate the response, whereas Dimension 1 has minimal impact. In Function 8, Dimensions 1, 3, and 7 must remain close to zero.
2. Mid-Level Interaction Motifs: Next, we modeled pairwise non-linear interactions across active dimensions. In Function 5, high outputs depend on the joint placement of Dimensions 3 and 4 in the upper corner above 0.95.
3. High-Level Global Acquisition: Finally, our acquisition function operates on the learned continuous manifold, balancing Expected Improvement across the global space with local ridge climbing.

### Question 2: Historical AI Breakthroughs vs Incremental Capstone Submissions
The historical breakthrough of AlexNet on ImageNet in 2012 resulted from three converging elements: accumulated data volume (ImageNet), expressive model capacity (deep CNNs), and efficient optimization (GPUs, ReLUs).

In our capstone trajectory:
1. Incubation Phase: Early rounds (Weeks 1 to 3) produced gradual progress as our surrogate models mapped sparse uncertainty across the search domain.
2. Breakthrough Phase: Once accumulated data reached critical density around promising basins, combining expressive non-linear surrogates with gradient-directed acquisition triggered dramatic performance leaps. In Function 5, steady exploratory queries (1088, then 2062, then 2304) culminated this round in a massive surge to y = 2921.37.

### Question 3: Depth and Complexity vs Exploration/Exploitation Trade-Offs
1. Deep Exploitation (Specialized Networks): Concentrating queries on known high-performing ridges yields rapid gains (such as pushing Function 5 coordinates to 0.907, 0.996, 0.944 for y = 2921.37). However, like overly deep networks that risk overfitting, excessive exploitation risks getting permanently trapped in a sub-optimal local mode.
2. Wide Exploration (Broad Generalization): Broad exploration samples unvisited quadrants with high model uncertainty, guaranteeing coverage and guarding against missed global optima, but consumes queries without immediately maximizing peak output.
We balanced this adaptively: allocating 75% of query focus to local exploitation on confirmed ascending ridges, and allocating query budget toward wide exploratory sampling on uncertain functions (Functions 1 and 3).

### Question 4: Neural Network Building Blocks & Model Learning
1. Loss Function and Target Scaling: Standard Mean Squared Error loss was disproportionately dominated by large-magnitude outputs (Function 5 with y above 2000), causing standard models to underfit lower-magnitude functions. Applying target normalization and standard scaling ensured balanced learning dynamics across all functions.
2. Non-Linear Activations (Threshold Behavior): Black-box functions frequently exhibit sharp threshold mechanics rather than smooth linear slopes. In Functions 4 and 5, the response surface remains flat until coordinates cross threshold boundaries (Dimension 3 exceeding 0.85), after which outputs change dramatically.
3. Gradients and Iterative Updates: Treating each weekly submission round as an iterative mini-batch update helped us view the optimization as a closed-loop trajectory: inputs produce evaluated outputs, prediction loss updates surrogate parameters, and output gradients point toward the next query location.

### Question 5: Framework Philosophy (PyTorch vs TensorFlow)
Our optimization framework represents an intentional hybrid:
1. Dynamic Flexibility (PyTorch): At the analytical level, we inspect individual function slice profiles, adjust exploration jitter based on empirical range, and evaluate multiple surrogate architectures interactively after each round.
2. Structured Production Design (TensorFlow): At the operational level, our pipeline (bbo_pipeline.py) standardizes data ingestion, runs automated 5-fold cross-validation, enforces unit hypercube constraints, serializes results to JSON, formats query strings, and automatically updates tracking dashboard visualisations.

### Question 6: Industry Applications (Giovanni Liotta - Sports AI) & Success Benchmarking
In professional sports analytics (player tracking, tactical evaluation, injury prevention), success is judged by decision utility under real-world uncertainty and physical constraints, rather than offline validation loss.

In our capstone:
1. Beyond Theoretical CV Score: A surrogate model with low Mean Squared Error is only an intermediate diagnostic. A model is only truly successful if its acquisition recommendations actively lead to higher observed function evaluations.
2. High-Stakes Decision Utility: With a strict budget of one evaluation per function per week, benchmarking success means evaluating how efficiently our framework converts sparse data points into record-breaking outputs.
