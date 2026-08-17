# Black-Box Optimization (BBO) - Week 5 (Module 16) Progress & Strategy Report
**Imperial College Capstone Project**

---

## 1. Week 5 Input Queries & Output Evaluation Summary

Following the evaluation of our Week 5 submissions, our optimization framework achieved another extraordinary milestone, unlocking a **massive all-time record global high on Function 5 (y = 2921.3747)**, while maintaining strong high-plateau performance on Functions 2 and 8.

### 1.1 Progression Table (Week 1 to Week 5)

| Function | Dim | Samples (N) | Previous Max y | W4 Evaluated y | Current Best y | Status / Gain | Week 5 Proposed Query Submission (x1-x2-...-xn) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| Func 1 | 2D | 14 | 7.71e-16 | 1.145e-65 | 7.71e-16 | Upper boundary mapped | 0.321608-0.165829 |
| Func 2 | 2D | 14 | 0.664342 | 0.656877 | 0.664342 | High plateau confirmed | 0.678392-0.879397 |
| Func 3 | 3D | 19 | -0.012927 | -0.125720 | -0.012927 | Explored corner mode | 0.455167-0.521242-0.486792 |
| Func 4 | 4D | 34 | 0.367529 | -1.514948 | 0.367529 | Valley edge bounded | 0.437086-0.290257-0.382594-0.349796 |
| Func 5 | 4D | 24 | 2304.288465 | 2921.374749 | 2921.374749 | Massive Record High (+617) | 0.595561-0.993735-0.987345-0.981682 |
| Func 6 | 5D | 24 | -0.305461 | -0.448204 | -0.305461 | Sub-space mapped | 0.437161-0.320742-0.578847-0.762968-0.194246 |
| Func 7 | 6D | 34 | 1.364968 | 0.899910 | 1.364968 | Clustered peak zone | 0.105640-0.347690-0.307992-0.359223-0.329529-0.769180 |
| Func 8 | 8D | 44 | 9.929387 | 9.613845 | 9.929387 | High ridge sustained | 0.050323-0.062907-0.187347-0.032471-0.743353-0.723330-0.136056-0.836079 |

---

## 2. Portal-Ready Reflection Questions and Answers (Module 16)

### Question 1
**Prompt**: How did the ideas of hierarchical feature learning influence the way you thought about structuring or refining your optimisation strategy this round?

**Answer**:
In deep learning, hierarchical feature learning describes how neural networks construct understanding in stages: early layers identify basic low-level features (such as edges and textures), middle layers assemble these into motif patterns, and deeper layers combine motifs into high-level holistic concepts.

This concept directly influenced how we structured our black-box optimization framework this round into a multi-tiered hierarchy:
1. Low-Level Dimensional Screening: First, we analyzed individual coordinate sensitivity to separate active driving dimensions from inactive nuisance parameters. For example, in Function 5, we identified that Dimension 3 and Dimension 4 dominate the response, whereas Dimension 1 has minimal standalone impact. In Function 8, we established that Dimensions 1, 3, and 7 must remain close to zero.
2. Mid-Level Interaction Motifs: Next, we modeled pairwise and non-linear interactions across active dimensions. In Function 5, high outputs do not depend on Dimension 3 or Dimension 4 in isolation, but on their simultaneous joint placement in the upper corner above 0.95.
3. High-Level Global Acquisition: Finally, our acquisition function operates on the learned continuous manifold, balancing Expected Improvement across the global space with local ridge climbing.

Structuring our reasoning hierarchically prevented us from treating all input dimensions equally, allowing us to focus our query budget strictly on high-yield coordinate subspaces.

---

### Question 2
**Prompt**: You saw how breakthroughs such as AlexNet and ImageNet classification reshaped expectations in AI. What parallels do you see between those leaps in performance and the incremental improvements you make in your capstone submissions?

**Answer**:
The historical breakthrough of AlexNet on ImageNet in 2012 was not a sudden accident; it occurred because three essential ingredients converged: a large accumulated volume of structured data (ImageNet), expressive non-linear model capacity (deep convolutional neural networks), and effective optimization (GPUs and ReLU activations).

We observed exact parallels in our capstone optimization trajectory:
1. The Incubation Phase (Incremental Progress): In early rounds (Weeks 1 to 3), progress appeared gradual as our surrogate models mapped sparse uncertainty across the domain. Evaluating points often returned modest incremental changes as we gathered foundational data points.
2. The Breakthrough Phase (Non-Linear Leap): Once accumulated samples reached a critical density around promising basins, combining expressive non-linear surrogates (Gaussian Processes and Neural Networks) with gradient-directed acquisition triggered dramatic performance leaps. This is vividly demonstrated in Function 5, where steady exploratory queries (outputs of 1088, then 2062, then 2304) culminated this round in a massive surge to y = 2921.37.

This parallel demonstrates that breakthrough performance in black-box optimization requires a patient initial phase of structured data gathering before high-yield exponential gains can be unlocked.

---

### Question 3
**Prompt**: When training neural networks, people often weigh trade-offs between depth, complexity and training efficiency. Did you encounter similar trade-offs in deciding whether to explore widely or exploit known promising regions in your queries?

**Answer**:
Yes, we encountered direct parallels to the depth versus complexity trade-off when allocating our weekly query budget between exploration and exploitation:

1. Deep Exploitation (Analogous to Deep, Specialized Networks):
Deeply exploiting a known high-performing region provides rapid, high-magnitude gains along established gradients. In Function 5, concentrating our queries on the upper ridge (pushing Dimension 2 to 0.907, Dimension 3 to 0.996, and Dimension 4 to 0.944) yielded our record output of 2921.37. However, just as overly deep networks risk overfitting and vanishing gradients, excessive exploitation risks getting permanently trapped in a sub-optimal local mode while ignoring undiscovered global peaks.

2. Wide Exploration (Analogous to Broad, Shallow Generalization):
Broad exploration samples unvisited quadrants with high model uncertainty, guaranteeing wide coverage and guarding against missed global optima. However, like shallow networks with limited capacity, exploration consumes precious weekly queries without immediately maximizing the top output score.

We resolved this trade-off adaptively: for functions with confirmed steep ascending gradients (such as Function 5), we allocated 75% of query focus to local exploitation. For multi-modal or flat functions (such as Function 1 and Function 3), we allocated query budget toward wide exploratory sampling in regions of maximum surrogate uncertainty.

---

### Question 4
**Prompt**: Reflecting on the building blocks of neural networks (inputs, activations, loss, gradients, weight updates), which of these concepts helped you think differently about how your model learns from the data you’ve accumulated so far?

**Answer**:
Three neural network building blocks fundamentally reshaped how we interpret our accumulated data:

1. Loss Function and Target Scaling:
Evaluating prediction errors revealed that standard Mean Squared Error loss was disproportionately dominated by large-magnitude outputs (such as Function 5 with y above 2000), causing standard models to underfit lower-magnitude functions. Applying target normalization and standard scaling ensured balanced learning dynamics across all functions.

2. Non-Linear Activations (Threshold Behavior):
Concepts like ReLU and Sigmoid activations illustrated that black-box functions frequently exhibit sharp threshold mechanics rather than smooth linear slopes. In Function 4 and Function 5, the response surface remains relatively flat until coordinates cross specific threshold boundaries (such as Dimension 3 exceeding 0.85), after which output values change dramatically. This highlighted why linear models failed and why non-linear surrogate flexibility was essential.

3. Gradients and Iterative Updates:
Treating each weekly submission round as an iterative mini-batch update helped us view the optimization as a closed-loop trajectory: inputs produce evaluated outputs, prediction loss updates surrogate model parameters, and output gradients point toward the next most promising query location.

---

### Question 5
**Prompt**: Module 16 also introduced PyTorch and TensorFlow as different frameworks for building and scaling models. If you were to frame your current optimisation approach in terms of a ‘framework’, would it be closer to rapid prototyping and flexibility or to structured, production-ready design? Why?

**Answer**:
Our optimization framework represents an intentional hybrid that evolved from PyTorch-style dynamic flexibility into a structured, production-ready design:

1. Dynamic Flexibility (PyTorch Philosophy):
At the analytical level, our strategy operates like PyTorch. Because each black-box function has distinct dimensionality (from 2D to 8D) and unique topological features, we dynamically inspect individual slice profiles, adjust exploration jitter based on empirical range, and evaluate multiple surrogate architectures (GPs, Neural Networks, Tree Ensembles) interactively after each round.

2. Structured Production Design (TensorFlow Philosophy):
At the operational level, our implementation follows a robust production architecture. Our automated pipeline (bbo_pipeline.py) standardizes data ingestion, runs automated 5-fold cross-validation, enforces strict boundary clipping in the unit hypercube, serializes weekly results to JSON, formats query strings to exact specifications, and automatically updates tracking dashboard visualisations.

This combination gives us the agility to tailor acquisition strategies to individual functions while ensuring 100% reliability, reproducibility, and error-free formatting for all portal submissions.

---

### Question 6
**Prompt**: In the guest interview, Giovanni Liotta discussed industry applications of deep learning in sport. How might reflecting on real-world deep learning use cases inform the way you benchmark success in your own capstone challenge?

**Answer**:
In Giovanni Liotta's discussion of deep learning in professional sports analytics (such as player tracking, tactical evaluation, and injury prevention), success is never defined merely by offline validation loss or theoretical model accuracy. Instead, success is judged by decision utility: does the model produce actionable, robust insights that improve performance under real-world uncertainty and physical constraints?

Reflecting on this real-world perspective reshaped how we benchmark success in our capstone optimization challenge:
1. Beyond Theoretical Cross-Validation Score: Having a surrogate model with a low Mean Squared Error is only an intermediate diagnostic. A model is only truly successful if its acquisition recommendations actively lead to higher observed function evaluations.
2. Decision Making Under Cost Constraints: In industrial AI and sports science, testing decisions in the real world is costly and limited. In our capstone, we have a strictly limited budget of one evaluation per function per week. Benchmarking success therefore means evaluating how efficiently our framework converts sparse data points into record-breaking outputs.

This alignment with real-world industry practices reinforces that predictive models are tools to guide high-stakes decision-making, where the ultimate benchmark of success is objective functional improvement.
