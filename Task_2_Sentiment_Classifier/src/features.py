"""
Feature Extraction Module for Sentiment Classification.

Provides TF-IDF Vectorization configuration and feature transformation pipelines.
Also documents the trade-offs between TF-IDF and Pretrained Dense Embeddings (e.g. BERT/Word2Vec).
"""

from typing import Tuple, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


# ---------------------------------------------------------------------------
# TRADEOFF ANALYSIS: TF-IDF vs Pre-trained Dense Embeddings (BERT / Word2Vec)
# ---------------------------------------------------------------------------
# 1. Computational Efficiency & Latency:
#    - TF-IDF: Extremely lightweight (O(N) token counting), trains in seconds on CPU,
#      requires virtually zero GPU compute, and has microsecond inference latency.
#    - Embeddings (BERT/RoBERTa): Heavy neural networks (100M+ parameters). Requires GPU
#      for fast inference, higher latency per sample (10-100x slower), and significantly
#      larger memory footprint.
#
# 2. Semantic & Contextual Understanding:
#    - TF-IDF: Bag-of-words / n-gram statistical approach. Disregards word order beyond
#      the specified n-gram window (e.g., bi-grams). Cannot inherently recognize synonyms
#      (e.g., 'fabulous' vs 'stellar') unless co-occurring in the vocabulary.
#    - Embeddings: Deep bidirectional self-attention mechanisms capture rich polysemy,
#      negation scope across long distances, and subtle semantic nuance.
#
# 3. Data Efficiency & Domain Specificity:
#    - TF-IDF: Outperforms heavy deep learning models on small to mid-sized datasets
#      (hundreds to thousands of samples) where large neural networks are prone to overfitting.
#    - Embeddings: Shines when fine-tuning on large corpora or leveraging massive pretraining
#      on general web knowledge.
#
# Verdict for this task:
# TF-IDF paired with n-grams (1, 2) and sublinear scaling provides an optimal balance:
# near-instant training/inference, zero GPU dependencies, excellent interpretability
# (interpretable feature weights/coefficients), and strong F1 accuracy on structured text.
# ---------------------------------------------------------------------------


class FeatureExtractor:
    """
    Manages text-to-numerical feature transformation using Scikit-Learn's TfidfVectorizer.
    """

    def __init__(
        self,
        ngram_range: Tuple[int, int] = (1, 2),
        max_features: Optional[int] = 5000,
        min_df: int = 2,
        max_df: float = 0.95,
        sublinear_tf: bool = True
    ):
        """
        Initialize the TF-IDF feature extractor.

        Args:
            ngram_range: Tuple (min_n, max_n) indicating unigrams, bigrams, etc.
            max_features: Maximum number of top vocabulary terms ordered by frequency.
            min_df: Minimum document frequency for a token to be considered.
            max_df: Maximum document frequency proportion (ignores corpus-wide stopwords).
            sublinear_tf: Apply sublinear scaling (1 + log(tf)) to prevent dominant terms.
        """
        self.vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            sublinear_tf=sublinear_tf,
            token_pattern=r"(?u)\b\w+\b"
        )
        self.is_fitted = False

    def fit_transform(self, texts):
        """
        Fit vectorizer vocabulary on training texts and return sparse TF-IDF matrix.
        """
        features = self.vectorizer.fit_transform(texts)
        self.is_fitted = True
        return features

    def transform(self, texts):
        """
        Transform unseen texts using the fitted vocabulary.
        """
        if not self.is_fitted:
            raise ValueError("FeatureExtractor must be fitted before transforming new data.")
        return self.vectorizer.transform(texts)

    def get_feature_names(self):
        """
        Return the list of vocabulary feature names.
        """
        if not self.is_fitted:
            raise ValueError("FeatureExtractor is not fitted.")
        return self.vectorizer.get_feature_names_out()


def explain_tradeoff() -> str:
    """
    Returns a formatted summary of the TF-IDF vs Pretrained Embedding tradeoffs.
    """
    return (
        "=== Tradeoff Analysis: TF-IDF vs Pre-trained Dense Embeddings ===\n"
        "1. Computation & Latency: TF-IDF is ultra-lightweight (CPU-friendly, sub-millisecond inference).\n"
        "   Transformer embeddings (e.g. BERT) require significant GPU resources and memory.\n"
        "2. Semantic Context: TF-IDF captures lexical frequencies and local n-grams (1, 2) but lacks\n"
        "   deep semantic synonymy. Transformers capture bidirectional contextual relationships.\n"
        "3. Small Dataset Suitability: For small-to-medium sentiment sets, TF-IDF + linear models\n"
        "   resist overfitting better, train instantly, and offer clear feature interpretability.\n"
    )
