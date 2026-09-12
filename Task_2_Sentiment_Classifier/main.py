"""
Main Pipeline Driver for Multi-Class Text Sentiment Classifier.

Orchestrates:
1. Dataset loading (or automated creation if not present)
2. Text preprocessing (cleaning, stop-words removal, POS-aware lemmatization)
3. Feature extraction (TF-IDF with n-grams and sublinear scaling)
4. Cross-validation model comparison (Logistic Regression, LinearSVC, Naive Bayes)
5. Model training and persistence
6. Comprehensive evaluation (Macro/Weighted F1, Precision, Recall, Confusion Matrix)
7. Sample inference demonstrations and interactive testing mode.
"""

import os
import sys
import argparse
from typing import Optional
import pandas as pd
from sklearn.model_selection import train_test_split

from src.preprocess import TextPreprocessor
from src.features import FeatureExtractor, explain_tradeoff
from src.train import (
    compare_models,
    train_classifier,
    SentimentPipeline
)
from src.evaluate import evaluate_model, print_evaluation_summary
from data.create_dataset import generate_curated_sentiment_dataset


def run_pipeline(
    data_path: str = "data/sentiment_data.csv",
    model_choice: str = "LogisticRegression",
    test_size: float = 0.20,
    random_state: int = 42,
    artifacts_dir: str = "artifacts"
) -> dict:
    """
    Executes the end-to-end sentiment classification pipeline.
    """
    print("\n" + "=" * 70)
    print("      TASK 2: MULTI-CLASS NATURAL LANGUAGE TEXT SENTIMENT CLASSIFIER")
    print("=" * 70)

    # 1. Dataset Acquisition
    if not os.path.exists(data_path):
        print(f"[Dataset] No dataset found at '{data_path}'. Generating curated benchmark dataset...")
        df = generate_curated_sentiment_dataset(data_path, samples_per_class=400)
    else:
        print(f"[Dataset] Loading existing dataset from '{data_path}'...")
        df = pd.read_csv(data_path)

    # Validate dataset structure
    if "text" not in df.columns or "sentiment" not in df.columns:
        raise ValueError("Dataset CSV must contain 'text' and 'sentiment' columns.")

    print(f"[Dataset] Total samples: {len(df)}")
    print(f"[Dataset] Class counts:\n{df['sentiment'].value_counts().to_string()}\n")

    # 2. Text Preprocessing
    print("[Preprocessing] Initializing TextPreprocessor (NLTK stop-words + POS lemmatization)...")
    preprocessor = TextPreprocessor(preserve_negations=True)
    
    # Preprocess all texts in the dataset
    print("[Preprocessing] Cleaning, tokenizing, filtering stopwords, and lemmatizing...")
    df["cleaned_text"] = preprocessor.preprocess_corpus(df["text"].tolist())
    
    # Remove any samples that became empty after cleaning
    initial_len = len(df)
    df = df[df["cleaned_text"].str.strip().str.len() > 0].reset_index(drop=True)
    if len(df) < initial_len:
        print(f"[Preprocessing] Removed {initial_len - len(df)} empty samples after preprocessing.")

    # 3. Stratified Train / Test Split
    print(f"\n[Split] Splitting data into train ({int((1-test_size)*100)}%) and test ({int(test_size*100)}%) with stratification...")
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        df["cleaned_text"].values,
        df["sentiment"].values,
        test_size=test_size,
        random_state=random_state,
        stratify=df["sentiment"].values
    )
    print(f"[Split] Training set: {len(X_train_raw)} samples | Test set: {len(X_test_raw)} samples")

    # 4. Feature Extraction (TF-IDF Vectorization)
    print("\n[Features] Extracting TF-IDF features (ngram_range=(1, 2), sublinear_tf=True)...")
    vectorizer = FeatureExtractor(
        ngram_range=(1, 2),
        max_features=5000,
        min_df=2,
        max_df=0.95,
        sublinear_tf=True
    )
    X_train_vec = vectorizer.fit_transform(X_train_raw)
    X_test_vec = vectorizer.transform(X_test_raw)
    print(f"[Features] Vocabulary size: {len(vectorizer.get_feature_names())} features")
    print(f"[Features] TF-IDF matrix shape: {X_train_vec.shape}")

    # Display Tradeoff analysis
    print("\n" + explain_tradeoff())

    # 5. Cross-Validation Model Comparison
    print("[Comparison] Evaluating candidate multi-class algorithms via 5-Fold Stratified CV...")
    comparison_results = compare_models(X_train_vec, y_train, cv_folds=5)

    # 6. Train Selected Classifier
    print(f"\n[Training] Training selected model: '{model_choice}' on full training set...")
    trained_model = train_classifier(X_train_vec, y_train, model_name=model_choice)
    print(f"[Training] Model '{model_choice}' trained successfully.")

    # 7. Model Evaluation on Held-Out Test Set
    print("\n[Evaluation] Evaluating model predictions on held-out test set...")
    y_pred = trained_model.predict(X_test_vec)
    
    unique_labels = sorted(list(set(y_train)))
    cm_plot_file = os.path.join(artifacts_dir, "confusion_matrix.png")
    
    metrics = evaluate_model(
        y_true=y_test,
        y_pred=y_pred,
        labels=unique_labels,
        save_cm_path=cm_plot_file
    )
    print_evaluation_summary(metrics)

    # 8. Persist End-to-End Pipeline Artifact
    pipeline = SentimentPipeline(
        preprocessor=preprocessor,
        vectorizer=vectorizer,
        model=trained_model
    )
    model_save_path = os.path.join(artifacts_dir, "sentiment_model.joblib")
    pipeline.save(model_save_path)

    # 9. Out-of-the-box Sample Inference Demo
    print("\n" + "-" * 70)
    print("                    LIVE INFERENCE DEMONSTRATION")
    print("-" * 70)
    demo_samples = [
        "Absolutely thrilled with this purchase! The customer support was top tier and incredibly helpful.",
        "The server returned HTTP status 200 and the database backup completed at 03:00 UTC.",
        "Worst experience ever. The application crashed and completely corrupted my saved files!",
        "The package arrived on Wednesday morning as per the standard delivery schedule.",
        "I was skeptical at first, but this tool completely transformed my daily workflow for the better.",
        "Terrible customer service, rude agent and they refused to honor the return policy.",
        "Please refer to paragraph 3 for the meeting agenda and venue details."
    ]

    preds = pipeline.predict(demo_samples)
    probas = pipeline.predict_proba(demo_samples)

    for i, (sample, pred) in enumerate(zip(demo_samples, preds)):
        conf_str = ""
        if probas is not None:
            max_p = probas[i].max() * 100
            conf_str = f" [Confidence: {max_p:.1f}%]"
        print(f"[{pred.upper():^8}]{conf_str} \"{sample}\"")
    print("-" * 70 + "\n")

    return {
        "metrics": metrics,
        "comparison": comparison_results,
        "pipeline": pipeline,
        "artifacts_dir": artifacts_dir
    }


def interactive_mode(pipeline_path: str = "artifacts/sentiment_model.joblib") -> None:
    """
    Launches an interactive command-line interface to test live sentences.
    """
    if not os.path.exists(pipeline_path):
        print(f"Error: Model pipeline artifact '{pipeline_path}' not found. Please train first.")
        return

    pipeline = SentimentPipeline.load(pipeline_path)
    print("\n" + "=" * 65)
    print("  INTERACTIVE SENTIMENT CLASSIFIER (Type 'exit' or 'quit' to stop)")
    print("=" * 65)

    while True:
        try:
            user_input = input("\nEnter text: ").strip()
            if user_input.lower() in ("exit", "quit"):
                print("Exiting interactive mode. Goodbye!")
                break
            if not user_input:
                continue

            pred = pipeline.predict([user_input])[0]
            probas = pipeline.predict_proba([user_input])

            print(f"-> Predicted Sentiment: {pred.upper()}")
            if probas is not None:
                prob_dict = {
                    cls: f"{p*100:.2f}%" for cls, p in zip(pipeline.classes_, probas[0])
                }
                print(f"-> Class Probabilities: {prob_dict}")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting interactive mode.")
            break


def main():
    parser = argparse.ArgumentParser(
        description="Multi-Class NLP Text Sentiment Classifier (Task 2)"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/sentiment_data.csv",
        help="Path to CSV dataset (must have 'text' and 'sentiment' columns)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="LogisticRegression",
        choices=["LogisticRegression", "LinearSVC", "MultinomialNB"],
        help="Model algorithm to train and evaluate"
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.20,
        help="Proportion of dataset held out for evaluation (default: 0.20)"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Launch interactive terminal inference mode after training"
    )

    args = parser.parse_args()

    results = run_pipeline(
        data_path=args.data_path,
        model_choice=args.model,
        test_size=args.test_size
    )

    if args.interactive:
        interactive_mode()


if __name__ == "__main__":
    main()
