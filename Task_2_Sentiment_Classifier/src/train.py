"""
Model Training & Selection Module for Multi-Class Text Sentiment Classifier.

Trains and evaluates candidate classifiers:
1. Multinomial Logistic Regression
2. Linear Support Vector Classifier (LinearSVC)
3. Multinomial Naive Bayes

Selects the best model according to Macro F1-score and exports the trained artifacts.
"""

import os
from typing import Dict, Any, Tuple, Optional
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_score, StratifiedKFold

from .features import FeatureExtractor
from .preprocess import TextPreprocessor


def build_candidate_models() -> Dict[str, Any]:
    """
    Constructs candidate multi-class classification models with tuned defaults.

    Returns:
        Dictionary mapping model names to Scikit-Learn estimator instances.
    """
    models = {
        # Logistic Regression with L2 regularization and balanced class weights
        "LogisticRegression": LogisticRegression(
            solver="lbfgs",
            class_weight="balanced",
            max_iter=1000,
            C=1.0,
            random_state=42
        ),
        # Linear SVM: effective for high-dimensional sparse text vectors
        "LinearSVC": CalibratedClassifierCV(
            LinearSVC(
                dual="auto",
                C=0.8,
                class_weight="balanced",
                random_state=42,
                max_iter=2000
            )
        ),
        # Multinomial Naive Bayes: classic probabilistic baseline
        "MultinomialNB": MultinomialNB(
            alpha=0.5,
            fit_prior=True
        )
    }
    return models


def compare_models(
    X_train_vec,
    y_train,
    cv_folds: int = 5
) -> Dict[str, Dict[str, float]]:
    """
    Performs Stratified K-Fold Cross-Validation across candidate models.

    Args:
        X_train_vec: Sparse TF-IDF feature matrix.
        y_train: Training labels array.
        cv_folds: Number of cross-validation splits.

    Returns:
        Summary dict of mean Macro F1, Weighted F1, and standard deviations per model.
    """
    models = build_candidate_models()
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    results = {}

    print(f"\nEvaluating candidate models via {cv_folds}-Fold Stratified Cross-Validation:")
    print("-" * 65)

    for name, model in models.items():
        macro_scores = cross_val_score(model, X_train_vec, y_train, cv=skf, scoring="f1_macro")
        weighted_scores = cross_val_score(model, X_train_vec, y_train, cv=skf, scoring="f1_weighted")

        results[name] = {
            "mean_macro_f1": float(np.mean(macro_scores)),
            "std_macro_f1": float(np.std(macro_scores)),
            "mean_weighted_f1": float(np.mean(weighted_scores)),
            "std_weighted_f1": float(np.std(weighted_scores)),
        }

        print(
            f"  {name:<20} | Macro F1: {results[name]['mean_macro_f1']:.4f} (+/- {results[name]['std_macro_f1']:.4f}) | "
            f"Weighted F1: {results[name]['mean_weighted_f1']:.4f}"
        )

    print("-" * 65)
    return results


def train_classifier(
    X_train_vec,
    y_train,
    model_name: str = "LogisticRegression"
):
    """
    Train a specific model on the complete training set.

    Args:
        X_train_vec: TF-IDF feature matrix for training.
        y_train: Training target labels.
        model_name: One of ['LogisticRegression', 'LinearSVC', 'MultinomialNB'].

    Returns:
        Fitted model estimator.
    """
    models = build_candidate_models()
    if model_name not in models:
        raise ValueError(f"Unknown model '{model_name}'. Choose from: {list(models.keys())}")

    model = models[model_name]
    model.fit(X_train_vec, y_train)
    return model


class SentimentPipeline:
    """
    End-to-end inference pipeline bundling preprocessor, vectorizer, and classifier.
    """

    def __init__(self, preprocessor: TextPreprocessor, vectorizer: FeatureExtractor, model: Any):
        self.preprocessor = preprocessor
        self.vectorizer = vectorizer
        self.model = model
        self.classes_ = getattr(model, "classes_", None)

    def predict(self, texts: list) -> list:
        """
        Preprocess, vectorize, and predict labels for an input list of raw texts.
        """
        cleaned = [self.preprocessor.preprocess_text(t) for t in texts]
        vecs = self.vectorizer.transform(cleaned)
        return self.model.predict(vecs).tolist()

    def predict_proba(self, texts: list) -> Optional[np.ndarray]:
        """
        Return predicted class probabilities if supported by the underlying model.
        """
        if not hasattr(self.model, "predict_proba"):
            return None
        cleaned = [self.preprocessor.preprocess_text(t) for t in texts]
        vecs = self.vectorizer.transform(cleaned)
        return self.model.predict_proba(vecs)

    def save(self, filepath: str) -> None:
        """
        Persist the trained pipeline to disk.
        """
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        joblib.dump(self, filepath)
        print(f"[Artifact Saved] Model pipeline serialized to -> {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "SentimentPipeline":
        """
        Load a persisted pipeline from disk.
        """
        return joblib.load(filepath)
