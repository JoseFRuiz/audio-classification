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

## 4. AUC loss

https://www.erikdrysdale.com/auc_max/

---

## 5. 🔹 **Evaluation Metrics: mAP and PR-AUC**

### **Average Precision (AP) and mean Average Precision (mAP)**

For multi-label classification, **Average Precision (AP)** measures the area under the **Precision-Recall curve** for each class independently. The **mean Average Precision (mAP)** is the average of AP across all classes.

For a single class $i$:

1. Sort all predictions by their predicted probability $p_i$ in descending order.
2. Compute precision and recall at each threshold:
   - **Precision** at threshold $t$: $P(t) = \frac{\text{TP}(t)}{\text{TP}(t) + \text{FP}(t)}$
   - **Recall** at threshold $t$: $R(t) = \frac{\text{TP}(t)}{\text{TP}(t) + \text{FN}(t)}$
3. The **AP** for class $i$ is the area under the Precision-Recall curve:
   $$
   \text{AP}_i = \int_0^1 P(R) \, dR
   $$
   In practice, this is approximated using the trapezoidal rule over discrete thresholds.

4. The **mAP** is the mean over all classes:
   $$
   \text{mAP} = \frac{1}{C} \sum_{i=1}^C \text{AP}_i
   $$

### **Precision-Recall AUC (PR-AUC)**

**PR-AUC** (Precision-Recall Area Under Curve) is the area under the Precision-Recall curve. For **multi-label classification**, PR-AUC is computed per class and then averaged, which is **equivalent to mAP**:

$$
\text{PR-AUC} = \frac{1}{C} \sum_{i=1}^C \int_0^1 P_i(R_i) \, dR_i = \text{mAP}
$$

**Key points:**
* **mAP = PR-AUC** for multi-label classification (they are the same metric).
* Both metrics are **threshold-independent** and evaluate ranking quality.
* They are particularly useful for **imbalanced datasets** where ROC-AUC can be misleading.
* Higher values indicate better performance (range: [0, 1]).

**Interpretation:**
* **mAP/PR-AUC = 1.0**: Perfect ranking (all positives ranked above all negatives).
* **mAP/PR-AUC = 0.5**: Random ranking.
* **mAP/PR-AUC < 0.5**: Worse than random (model is systematically wrong).

---

## Summary Table

| Loss     | Formula                                                               | Highlights                                                |
| -------- | --------------------------------------------------------------------- | --------------------------------------------------------- |
| **BCE**  | $-\sum_i \left[ y_i \log p_i + (1 - y_i) \log (1 - p_i) \right]$      | Treats all labels equally                                 |
| **ASL**  | Weighted BCE with asymmetric focusing and margin                      | Focuses on hard positives and suppresses easy negatives   |
| **MCRL** | $\max\left(0, \delta - (\mu_{\text{pos}} - \mu_{\text{neg}}) \right)$ | Enforces average score of positives to be above negatives |


### Classifiers with Wav2Vec

| 📘 **Paper & Link**                                                                                                                                                                      | 🎯 **Task**                               | 🧠 **Architecture**                                     |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------- | ------------------------------------------------------- |
| **Improving Tone Recognition Performance using Wav2Vec 2.0**<br>(Yoruba tone recognition) — **GRU & Light‑GRU**<br>[Link](https://dl.acm.org/doi/10.1145/3690384?utm_source=chatgpt.com) | Tone recognition (low‑resourced language) | GRU / LiGRU                                             |
| **Emotion Recognition from Speech Using Wav2Vec 2.0 Embeddings**<br>(Pepino et al., Interspeech 2021) — **MLP / LSTM fusion**<br>[ArXiv/ISCA](https://arxiv.org/abs/2104.03502)          | Speech emotion (IEMOCAP, RAVDESS)         | Multiple shallow models including LSTM                  |
| **Speech Emotion Recognition using fine‑tuned Wav2Vec 2.0 + NCDE**<br>(Wang & Yang, PLoS ONE 2025) — **NCDE**<br>[PLOS ONE](https://doi.org/10.1371/journal.pone.0318297)                | Speech emotion (IEMOCAP)                  | Neural Controlled Differential Equations                |
| **Exploring Wav2Vec 2.0 fine‑tuning for improved SER**<br>(Chen & Rudnicky, ArXiv 2021) — **Transformers**<br>[ArXiv](https://arxiv.org/abs/2110.06309)                                  | SER (IEMOCAP)                             | Fine-tuned Wav2Vec + Transformer fine-tuning strategies |
| **Multi-level Fusion of Wav2Vec 2.0 and BERT for Multimodal Emotion Recognition**<br>(Zhao et al., ArXiv 2022) — **Co-Attention**<br>[ArXiv](https://arxiv.org/abs/2207.04697)           | Multimodal emotion recognition            | Co-attention Fusion with Wav2Vec and BERT               |
| **Dawn of the Transformer Era in Speech Emotion Recognition**<br>(Wagner et al., T-PAMI 2023) — **Audio Transformer**<br>[Zenodo model](https://zenodo.org/record/6221127)               | Dimensional emotion (valence/arousal)     | Fine-tuned Wav2Vec + Transformer head                   |
