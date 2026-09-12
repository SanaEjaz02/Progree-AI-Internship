"""
Unit & Integration Tests for Task 2 Sentiment Classifier Components.
"""

import os
import unittest
import numpy as np

from src.preprocess import TextPreprocessor
from src.features import FeatureExtractor
from src.train import train_classifier, SentimentPipeline
from src.evaluate import evaluate_model
from data.create_dataset import generate_curated_sentiment_dataset


class TestSentimentClassifier(unittest.TestCase):

    def setUp(self):
        self.preprocessor = TextPreprocessor(preserve_negations=True)
        self.vectorizer = FeatureExtractor(ngram_range=(1, 2), min_df=1)

    def test_preprocessing_negation_retention(self):
        # 'not' should be retained because preserve_negations=True
        text = "This is not good at all!"
        cleaned = self.preprocessor.preprocess_text(text)
        self.assertIn("not", cleaned)

    def test_preprocessing_clean_urls_and_mentions(self):
        text = "Check out https://google.com @user amazing app!"
        cleaned = self.preprocessor.preprocess_text(text)
        self.assertNotIn("https", cleaned)
        self.assertNotIn("@user", cleaned)
        self.assertIn("amaze", cleaned)  # Lemmatized from 'amazing'

    def test_feature_extraction(self):
        corpus = [
            "customer support be great",
            "terrible slow service not work",
            "meeting scheduled for monday"
        ]
        X = self.vectorizer.fit_transform(corpus)
        self.assertEqual(X.shape[0], 3)
        self.assertTrue(X.shape[1] > 0)

    def test_training_and_pipeline(self):
        corpus = [
            "fantastic experience love it",
            "great service awesome",
            "terrible slow broken crash",
            "worst product ever hate it",
            "office open at nine am",
            "scheduled flight departure time"
        ]
        labels = ["positive", "positive", "negative", "negative", "neutral", "neutral"]
        
        preprocessed = self.preprocessor.preprocess_corpus(corpus)
        X_vec = self.vectorizer.fit_transform(preprocessed)
        model = train_classifier(X_vec, labels, model_name="LogisticRegression")
        
        pipeline = SentimentPipeline(self.preprocessor, self.vectorizer, model)
        preds = pipeline.predict(["I love this, great job!", "Completely broken and terrible"])
        self.assertEqual(len(preds), 2)
        self.assertEqual(preds[0], "positive")
        self.assertEqual(preds[1], "negative")

    def test_evaluation_metrics(self):
        y_true = ["positive", "neutral", "negative", "positive"]
        y_pred = ["positive", "neutral", "negative", "negative"]
        metrics = evaluate_model(y_true, y_pred)
        self.assertIn("macro_f1", metrics)
        self.assertIn("weighted_f1", metrics)
        self.assertIn("confusion_matrix", metrics)
        self.assertTrue(0.0 <= metrics["macro_f1"] <= 1.0)


if __name__ == "__main__":
    unittest.main()
