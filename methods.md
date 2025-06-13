## Notation

Let:

* $\mathbf{x} \in \mathbb{R}^C$ be the **logits** (raw outputs) for $C$ classes.
* $\mathbf{p} = \sigma(\mathbf{x}) \in (0,1)^C$ be the **sigmoid probabilities**, where $\sigma(x) = \frac{1}{1 + e^{-x}}$.
* $\mathbf{y} \in \{0,1\}^C$ be the **ground truth binary label vector** (multi-label).
* Let $\mathcal{P} = \{ i \mid y_i = 1 \}$ denote the set of **positive label indices**.
* Let $\mathcal{N} = \{ i \mid y_i = 0 \}$ denote the set of **negative label indices**.

---

## 1. 🔹 **Binary Cross-Entropy Loss (BCE)**

The BCE loss for multi-label classification is computed **independently for each class**:

$$
\mathcal{L}_{\text{BCE}}(\mathbf{p}, \mathbf{y}) = -\sum_{i=1}^C \left[ y_i \log(p_i) + (1 - y_i) \log(1 - p_i) \right]
$$

---

## 2. 🔹 **Asymmetric Loss (ASL)**

ASL modifies BCE by applying **different focusing factors** for positive and negative labels, and optionally includes a **margin** to suppress easy negatives.

Let:

* $\gamma^+$ be the focusing parameter for **positives**,
* $\gamma^-$ be the focusing parameter for **negatives**,
* $m$ be a margin that shifts negative probabilities.

Then:

$$
\mathcal{L}_{\text{ASL}}(\mathbf{p}, \mathbf{y}) = - \sum_{i \in \mathcal{P}} (1 - p_i)^{\gamma^+} \log(p_i)
\quad
- \sum_{j \in \mathcal{N}} \left( \max(p_j - m, 0) \right)^{\gamma^-} \log(1 - \max(p_j - m, 0))
$$

* The margin $m \in [0, 1)$ helps **ignore easy negatives**.
* The powers $\gamma^+$, $\gamma^-$ help focus learning on **hard examples**.

---

## 3. 🔹 **Mean Contrastive Ranking Loss (MCRL)**

This loss encourages the **mean predicted score** for the positive classes to exceed that of the negative classes by a fixed **margin** $\delta$. It is defined as:

$$
\mathcal{L}_{\text{MCRL}}(\mathbf{p}, \mathbf{y}) = \max \left( 0,\ \delta - \left( \frac{1}{|\mathcal{P}|} \sum_{i \in \mathcal{P}} p_i - \frac{1}{|\mathcal{N}|} \sum_{j \in \mathcal{N}} p_j \right) \right)
$$

* $\delta$ is a tunable margin hyperparameter.
* This loss applies a **ranking constraint** at the aggregate level, rather than per pair.

---

## Summary Table

| Loss     | Formula                                                               | Highlights                                                |
| -------- | --------------------------------------------------------------------- | --------------------------------------------------------- |
| **BCE**  | $-\sum_i \left[ y_i \log p_i + (1 - y_i) \log (1 - p_i) \right]$      | Treats all labels equally                                 |
| **ASL**  | Weighted BCE with asymmetric focusing and margin                      | Focuses on hard positives and suppresses easy negatives   |
| **MCRL** | $\max\left(0, \delta - (\mu_{\text{pos}} - \mu_{\text{neg}}) \right)$ | Enforces average score of positives to be above negatives |