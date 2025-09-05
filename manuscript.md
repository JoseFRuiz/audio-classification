# Understanding the Role of Loss Functions in Multi-Label Audio Classification with Learned Features

Loss functions tailored for multi-label classification (such as asymmetric formulations and AUC-optimized surrogates) are often proposed to handle label imbalance and ranking-based evaluation metrics. In this work, we revisit this design space by empirically comparing three loss functions: Binary Cross-Entropy (BCE), the Asymmetric Loss Function (ASL), and a recent surrogate that directly optimizes Macro-AUC. Experiments are conducted on two challenging audio datasets, FSD50K and BirdCLEF, using a Wav2Vec feature extractor followed by a GRU-based classifier.

Despite lacking explicit mechanisms for imbalance or AUC optimization, BCE consistently outperforms or matches more specialized alternatives. On FSD50K, BCE achieves the highest validation AUC (0.883), surpassing ASL (0.862), the AUC-optimized surrogate (0.853), and all linear combinations thereof. Similar patterns are observed on BirdCLEF. Additional experiments replacing Wav2Vec with raw waveform inputs confirm that better training loss does not imply better generalization.

These findings suggest that when paired with strong pre-trained representations, BCE offers competitive optimization stability and generalization, challenging assumptions about the necessity of complex loss functions in modern multi-label pipelines.
