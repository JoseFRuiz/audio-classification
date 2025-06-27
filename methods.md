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
