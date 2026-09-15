# Black-Box Optimization (BBO) Weekly Tracking Log
**Imperial College Capstone Project**

This document serves as the master tracking log for optimising the 8 synthetic black-box functions. It documents our framework design, weekly surrogate model cross-validation benchmarks, portal submission strings, visual diagnostic progress, and formal reflections across all rounds.

---

## 1. Master Dashboard & Weekly Optimization Trajectory

![Weekly Progress Master Dashboard](file:///c:/Users/sesa625752/OneDrive%20-%20Schneider%20Electric/Imperial_College/Capstone_Project/visualizations/weekly_progress_summary.png)

The master dashboard image above ([weekly_progress_summary.png](file:///c:/Users/sesa625752/OneDrive%20-%20Schneider%20Electric/Imperial_College/Capstone_Project/visualizations/weekly_progress_summary.png)) provides a 4-panel overview:
1. Current Max vs Predicted Next Query Output: Side-by-side comparison of current maximum y vs GP mean prediction with error bars for Functions 1 to 8.
2. Weekly Optimization Trajectory: Historical line plot tracking peak output achieved per function across weekly rounds (Week 1 to Week 8).
3. Surrogate Model CV Performance: Bar chart illustrating the winning surrogate model and 5-Fold CV-MSE score across all 8 functions.
4. Expected Improvement (EI) Potential Ratio: Normalized gain metric highlighting which functions possess the highest potential for global peak discovery in upcoming submissions.

---

## 2. Multi-Week Progress & Week 8 Submissions Summary

### 2.1 Function Progression & Week 8 Submissions Table

| Function | Dim | Initial to Current Samples | Previous Max y | W7 Evaluated y | Current Best y | Status / Gain | Week 8 Proposed Query Submission (x1-x2-...-xn) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| Func 1 | 2D | 10 to 17 | 7.71e-16 | -7.680e-87 | 7.71e-16 | Upper boundary mapped | 0.467337-0.899497 |
| Func 2 | 2D | 10 to 17 | 0.664342 | 0.261147 | 0.664342 | High plateau confirmed | 0.708543-0.899497 |
| Func 3 | 3D | 15 to 22 | -0.004807 | -0.109442 | -0.004807 | Central mode bounded | 0.001199-0.001794-0.625666 |
| Func 4 | 4D | 30 to 37 | 0.367529 | -36.647771 | 0.367529 | Valley edge bounded | 0.382449-0.359348-0.511752-0.380244 |
| Func 5 | 4D | 20 to 27 | 5893.206247 | 4416.088821 | 5893.206247 | Astronomical Peak Region | 0.922756-0.986637-0.984891-0.895457 |
| Func 6 | 5D | 20 to 27 | -0.216645 | -0.472698 | -0.216645 | High plateau mapped | 0.455167-0.358085-0.501329-0.612289-0.118077 |
| Func 7 | 6D | 30 to 37 | 2.233318 | 2.043269 | 2.233318 | High basin sustained | 0.004479-0.306936-0.503959-0.377022-0.307476-0.830835 |
| Func 8 | 8D | 40 to 47 | 9.929387 | 9.951309 | 9.951309 | All-Time Peak Surge | 0.064744-0.079547-0.264656-0.342390-0.954529-0.992384-0.077378-0.241817 |

---

## 3. Week 8 (Module 19) Reflection & Strategy Report (Portal-Ready)

### Question 1: Prompt Patterns (Zero-Shot vs Few-Shot)
We implemented a Few-Shot Structured In-Context Learning Prompt Pattern. In black-box optimization, zero-shot prompting leads to unconstrained hallucinations, such as generating out-of-bound coordinates outside the unit hypercube (0, 1) or malformed delimiters. Few-shot exemplars provide explicit in-context demonstrations of historical input-output pairs across sequential rounds, constraining the model's structural attention.

Simplifying prompts (such as requesting a "good query for Function 5") caused formatting drift, missing dimensions, and unguided coordinate guessing. Structuring prompts with explicit System Roles, XML field delimiters, and 5 historical input-output exemplars forced strict schema compliance (x1-x2-...-xn) and focused reasoning on empirical trend gradients.

### Question 2: Decoding Parameters
We selected a low-temperature, constrained decoding configuration:
1. Temperature = 0.2: Low temperature minimized stochastic sampling noise. High temperature (>0.8) introduced chaotic coordinate jitter, risking query budget waste. Low temperature ensured deterministic, coherent compliance with mathematical constraints.
2. Top-p (Nucleus Sampling) = 0.95: Excluded improbable low-probability tokens while preserving minor coordinate diversity near high-yield ridges.
3. Top-k = 40: Restricted candidate token vocabulary strictly to numeric digits and standard string delimiters.
4. Max-tokens = 1000: Prevented premature output truncation during multi-step cross-validation reasoning and query string formatting.

Low temperature trading off diversity for strict mathematical coherence enabled our framework to exploit Function 8's active subspace, unlocking an all-time record peak of y = 9.951309 this round.

### Question 3: Tokenization & String Safeguards
Floating-point numeric strings (such as 0.969673) are split by sub-word tokenizers into arbitrary byte-pair tokens (like "0.", "969", "673"). Standard LLM tokenizers often mangle multi-digit floating-point values or strip bracketed arrays during text processing.

Safeguards and Verification: To prevent tokenization artifacts, we enforced standardized 6-decimal string formatting (f"{x:.6f}") and substituted hyphenated plain text delimiters (0.xxxxxx-0.yyyyyy) for bracketed arrays. We validated string parsing using automated regex checks prior to portal submission, confirming zero truncation or token-mangling failures across all 8 functions.

### Question 4: Limitations at 17+ Data Points
As evaluation datasets expanded to 17+ points per function (reaching 47 points in Function 8), raw text context prompts triggered clear attentional limitations:
1. Attention Fragmentation ("Lost in the Middle"): Passing long unformatted point histories caused LLM attention mechanisms to over-weight early initial points or recent queries while ignoring intermediate gradient trends.
2. Prompt Overfitting and Diminishing Returns: Appending raw text logs increased context length without improving decision quality. We eliminated prompt overfitting by passing pre-computed statistical summaries—including top-performing coordinates, 5-fold CV model rankings, and Gini feature importances—reducing context noise and maximizing attention efficiency.

### Question 5: Anti-Hallucination Strategies
We deployed a three-tier anti-hallucination defense:
1. Tighter System Instructions: Declared explicit numerical boundaries (0.000000 <= x_i <= 1.000000) and zero-tolerance rules for out-of-bound values.
2. Retrieval-Augmented Telemetry (RAG): Ingested exact NumPy arrays, 5-fold cross-validation MSE scores, and range-scaled expected improvement values directly into context.
3. Constrained Output Schema: Enforced serialized JSON outputs and exact string templates (x1-x2-...-xn).

### Question 6: Scaling Prompting & Decoding
To scale prompting and decoding for larger datasets (100+ points) and complex models:
1. k-NN Vector Retrieval RAG: Rather than dumping full text history, we will retrieve only the k-nearest historical evaluations surrounding the target query candidate region.
2. LLM Tool-Calling Execution: We will transition from predicting numeric text strings directly to LLM Tool-Calling (Function Calling), allowing the LLM to write and execute Python scikit-learn optimization scripts deterministically.

### Question 7: Practitioner Mindset
Professional AI practitioners recognize that LLMs are non-deterministic reasoning engines requiring strict deterministic guardrails. Wrapping low-temperature decoding within structured retrieval telemetry taught us to treat LLMs as orchestrators rather than numeric calculators, balancing high-yield exploitation against exploratory risk under strict weekly query limits.

### Question 8: Synthesis (Structured Prompts vs Exploration)
Structured prompts paired with Gaussian Process cross-validation significantly reduce epistemic uncertainty near confirmed peaks, fully justifying deep exploitation along high-yield ridges—as demonstrated by Function 5 reaching 5893.21 and Function 8 achieving its new record of 9.951309.

However, because finite attention windows can suffer from context fragmentation and sub-word tokenization can distort floating-point precision, pure exploitation is risky. We maintain active exploration on uncertain functions (Functions 1 and 4) using range-scaled acquisition jitter. At 17+ data points, raw context logs show clear diminishing returns; replacing raw logs with pre-computed surrogate telemetry completely resolves prompt overfitting and attention degradation.
