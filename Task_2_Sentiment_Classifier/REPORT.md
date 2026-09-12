# Task 2: Intelligent Multi-Class Natural Language Text Sentiment Classifier
**Internship Technical Report & Model Whitepaper**

---

## 1. Executive Summary

This report documents the design, implementation, and evaluation of **Task 2: Intelligent Multi-Class Natural Language Text Sentiment Classifier** for the Progree Artificial Intelligence Internship. 

The objective was to engineer an NLP classification system capable of mapping and categorizing unstructured text statements into three distinct sentiment classes: **Positive**, **Neutral**, and **Negative**. The system features an end-to-end modular pipeline consisting of surface text normalization, negation-aware stop-words filtering, Part-of-Speech (POS) guided lemmatization, bi-gram TF-IDF feature extraction, cross-validated multi-class model benchmarking, and full diagnostic evaluation.

---

## 2. Model Selection & Architecture Justification

### Why Scikit-Learn over PyTorch / Heavy Deep Learning?
For short-to-medium length customer reviews, support queries, and operational updates on tabular text datasets, **Scikit-Learn linear and probabilistic classifiers** offer major advantages over heavyweight PyTorch neural networks:
1. **Sample Efficiency**: Deep neural networks (e.g., LSTMs, Transformers) require massive corpora (tens of thousands of samples) to avoid severe overfitting. Linear models paired with convex loss functions generalize effectively on small-to-medium datasets.
2. **Computational Overhead & Latency**: Scikit-Learn classifiers train in under 2 seconds on standard CPU hardware with zero GPU dependencies, while achieving sub-millisecond inference latencies.
3. **Calibrated Probabilities & Interpretability**: Multinomial Logistic Regression provides calibrated posterior probabilities $P(Y=c \mid X)$ via the Softmax function and directly inspectable feature coefficients, enabling error analysis and auditing.

### Algorithm Comparison (5-Fold Stratified Cross-Validation)
Three algorithms were benchmarked on the training set using 5-Fold Stratified Cross-Validation:

| Model Architecture | Mean Macro F1-Score | Std Dev ($\pm$) | Mean Weighted F1-Score | Key Strength |
| :--- | :---: | :---: | :---: | :--- |
| **Multinomial Logistic Regression** | **0.9761** | $\pm 0.0078$ | **0.9761** | Smooth $L_2$ regularization, calibrated probabilities |
| **Linear Support Vector Machine (LinearSVC)** | 0.9751 | $\pm 0.0068$ | 0.9751 | High-dimensional margin maximization |
| **Multinomial Naive Bayes** | 0.9771 | $\pm 0.0078$ | 0.9771 | Fast probabilistic frequency baseline |

**Selected Production Model**: **Multinomial Logistic Regression** (`solver='lbfgs'`, `class_weight='balanced'`, $C=1.0$). While all three models achieved competitive F1 scores, Logistic Regression was selected because it naturally outputs calibrated confidence distributions across classes (essential for production thresholding) and handles class weighting smoothly.

---

## 3. Preprocessing Pipeline

The preprocessing stage transforms raw, noisy text into clean linguistic tokens while preserving semantic intent:

```
Raw Text 
  └──> 1. Surface Cleaning (lower casing, URLs, HTML tags, @mentions, non-alpha symbols)
  └──> 2. Tokenization (NLTK punkt tokenizer)
  └──> 3. Negation-Aware Stop-Words Filtering (removes filler words, retains 'not', 'no', 'never')
  └──> 4. POS-Guided WordNet Lemmatization (nouns, verbs, adjectives, adverbs)
  └──> Normalized Lemmatized String
```

### Critical Design Decision: Retaining Sentiment Negations
Standard stop-word lists filter out words like *"not"*, *"no"*, *"never"*, *"barely"*, and *"hardly"*. In sentiment analysis, removing these tokens causes catastrophic sentiment inversion (e.g., *"not happy"* becomes *"happy"*). 

Our `TextPreprocessor` incorporates a guarded exclusion set (`CRITICAL_NEGATIONS`) that preserves negation modifiers while stripping standard functional words (*"the"*, *"in"*, *"at"*, *"which"*, etc.).

### Part-of-Speech (POS) Guided Lemmatization
Standard lemmatizers default to treating all tokens as nouns (`n`). Consequently:
- *"running"* remains *"running"* instead of verb lemma *"run"*.
- *"better"* remains *"better"* instead of adjective lemma *"good"*.

Our pipeline tags tokens using Penn Treebank POS tags and maps them dynamically to WordNet POS constants (`ADJ`, `VERB`, `NOUN`, `ADV`), ensuring optimal morphological normalization.

---

## 4. Feature Extraction & Embedding Tradeoff Analysis

### TF-IDF Configuration
Text tokens are mapped to high-dimensional numerical features using Scikit-Learn's `TfidfVectorizer` configured with:
- **Bi-grams** (`ngram_range=(1, 2)`): Captures contiguous word pairs to recognize compound phrases like *"not bad"*, *"slow service"*, *"great job"*.
- **Sublinear Term Frequency** (`sublinear_tf=True`): Replaces raw term frequency $tf$ with $1 + \log(tf)$ to dampen the impact of repeatedly occurring tokens.
- **Frequency Filtering** (`min_df=2`, `max_df=0.95`): Filters out corpus-wide ubiquitous words and single-instance typos.

### Tradeoff Analysis: TF-IDF vs Pretrained Dense Embeddings (BERT / Transformers)

| Evaluation Dimension | TF-IDF Vectorization | Pretrained Dense Embeddings (e.g., BERT / RoBERTa) |
| :--- | :--- | :--- |
| **Compute & Hardware** | **Extremely Lightweight**: CPU-only, minimal RAM, instant vectorization. | **Heavy**: Demands GPU/TPU accelerators and substantial VRAM (hundreds of MBs per model). |
| **Latency** | **Sub-millisecond**: $\sim 0.1 \text{ ms}$ per sample; ideal for high-throughput streaming. | **Moderate to High**: $15 \text{ ms} - 100 \text{ ms}$ per inference turn. |
| **Semantic Synonyms** | **Lexical only**: Treats *"fabulous"* and *"stellar"* as orthogonal features unless co-trained. | **Deep Contextual**: Maps synonyms to proximal vectors in latent space. |
| **Negation & Word Order** | **Local**: Captures fixed $n$-grams (e.g., bi-grams), but misses long-range syntactic shifts. | **Global**: Bidirectional self-attention tracks dependencies across entire passages. |
| **Sample Efficiency** | **High**: Resists overfitting on smaller datasets (hundreds to thousands of samples). | **Prone to Overfitting**: Fine-tuning requires large annotated datasets or parameter-efficient adapters (LoRA). |
| **Interpretability** | **Glass-Box**: Feature weights directly map to explicit words and bi-grams. | **Black-Box**: Requires complex attribution methods (Integrated Gradients, SHAP). |

**Conclusion**: For real-time, low-latency, and cost-effective text sentiment classification on standard structured texts, **TF-IDF with bi-grams represents the optimal engineering tradeoff**. When semantic synonymy across open-domain dialogue or complex sentence structures is paramount, migrating to a distilled Transformer (e.g., `DistilBERT` or `MiniLM`) is the logical progression.

---

## 5. Quantitative Evaluation & Benchmark Results

The model was evaluated on a held-out test split ($20\%$, $N=240$ samples) with stratified class representation ($80$ samples per class).

### Primary Metrics Summary

| Evaluation Metric | Test Set Score | Description |
| :--- | :---: | :--- |
| **Overall Accuracy** | **97.50%** | Overall percentage of correct predictions |
| **Macro F1-Score** | **0.9749** | Unweighted average of F1 across classes (evaluates class balance) |
| **Weighted F1-Score** | **0.9749** | F1 weighted by support size |
| **Macro Precision** | **0.9750** | Unweighted mean precision across classes |
| **Macro Recall** | **0.9750** | Unweighted mean recall across classes |

### Per-Class Performance (Classification Report)

| Sentiment Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Negative** | 0.97 | 0.95 | **0.96** | 80 |
| **Neutral** | 0.98 | 1.00 | **0.99** | 80 |
| **Positive** | 0.97 | 0.97 | **0.97** | 80 |
| **Macro Average** | **0.97** | **0.97** | **0.97** | 240 |
| **Weighted Average** | **0.97** | **0.97** | **0.97** | 240 |

### Confusion Matrix Analysis

```
Actual \ Predicted       Negative      Neutral      Positive
------------------------------------------------------------
Negative                   76            2            2
Neutral                     0           80            0
Positive                    2            0           78
```

- **Neutral Class**: Achieved **100% recall** ($80/80$ correct), with zero neutral statements misclassified as emotional polarities.
- **Negative Class**: $76/80$ correct ($95\%$ recall). Misclassifications involved subtle multi-clause constructs.
- **Positive Class**: $78/80$ correct ($97\%$ recall).

The confusion matrix visualization is saved at:  
`artifacts/confusion_matrix.png`

---

## 6. Live Inference Validation

The serialized pipeline was validated on unseen real-world statements:

| Input Text | Predicted Class | Confidence Score | Result |
| :--- | :---: | :---: | :---: |
| *"Absolutely thrilled with this purchase! The customer support was top tier and incredibly helpful."* | `POSITIVE` | 44.5% | Correct |
| *"The server returned HTTP status 200 and the database backup completed at 03:00 UTC."* | `NEUTRAL` | 63.7% | Correct |
| *"Worst experience ever. The application crashed and completely corrupted my saved files!"* | `NEGATIVE` | 57.6% | Correct |
| *"The package arrived on Wednesday morning as per the standard delivery schedule."* | `NEUTRAL` | 80.6% | Correct |
| *"I was skeptical at first, but this tool completely transformed my daily workflow for the better."* | `POSITIVE` | 55.1% | Correct |
| *"Terrible customer service, rude agent and they refused to honor the return policy."* | `NEGATIVE` | 51.9% | Correct |
| *"Please refer to paragraph 3 for the meeting agenda and venue details."* | `NEUTRAL` | 75.6% | Correct |

---

## 7. How to Run & Reproduce

All commands can be executed from the project directory:

```bash
cd "Task_2_Sentiment_Classifier"

# 1. Run full pipeline (training, cross-validation, evaluation, demo)
python main.py

# 2. Train with a specific algorithm (LinearSVC or MultinomialNB)
python main.py --model LinearSVC

# 3. Supply a custom CSV dataset
python main.py --data-path "path/to/custom_data.csv"

# 4. Launch interactive terminal inference loop
python main.py --interactive

# 5. Run test suite
python -m unittest tests/test_components.py
```

---

## 8. Conclusion

The developed **Multi-Class Natural Language Text Sentiment Classifier** fulfills all requirements of **Task 2**:
- Robust text normalization with negation-preserving stop-words removal and POS-guided lemmatization.
- Tuned TF-IDF feature extraction with bi-grams and sublinear scaling.
- Multi-model cross-validation and selection of Multinomial Logistic Regression.
- Rigorous metric reporting (**0.9749 Macro F1**, **0.9749 Weighted F1**, full precision/recall report, and confusion matrix heatmap).
- Clean, production-ready, modular architecture with automated test coverage and interactive inference capabilities.
