"""
Text Preprocessing Module for Multi-Class Sentiment Classifier.

Handles text normalization, stop-words removal (preserving key negation tokens),
and Part-of-Speech (POS) guided lemmatization using NLTK.
"""

import re
import string
from typing import List, Optional
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, word_tokenize


def ensure_nltk_resources() -> None:
    """
    Ensure required NLTK corpora and models are available.
    Only downloads if missing locally.
    """
    packages = [
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
    ]
    for resource_path, pkg_name in packages:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            try:
                nltk.download(pkg_name, quiet=True)
            except Exception:
                pass



class TextPreprocessor:
    """
    Configurable text preprocessor providing normalization, stop-word removal,
    and POS-tagged WordNet lemmatization.
    """

    # Sentiment-critical negation tokens that should NOT be removed by stopwords
    CRITICAL_NEGATIONS = {
        "not", "no", "nor", "neither", "never", "none", "hardly", "scarcely",
        "barely", "ain", "aren", "couldn", "didn", "doesn", "don", "hadn",
        "hasn", "haven", "isn", "mightn", "mustn", "needn", "shan", "shouldn",
        "wasn", "weren", "won", "wouldn", "without", "against"
    }

    def __init__(self, preserve_negations: bool = True):
        """
        Initialize the preprocessor and configure stop-word lists and lemmatizer.

        Args:
            preserve_negations (bool): If True, retains negation words in the stop-words filter
                                      to prevent sentiment inversion (e.g. 'not bad' != 'bad').
        """
        ensure_nltk_resources()
        self.lemmatizer = WordNetLemmatizer()
        
        # Build stop-words set from NLTK English stopwords
        try:
            raw_stops = set(stopwords.words("english"))
        except Exception:
            # Fallback set if NLTK stopwords download fails offline
            raw_stops = {
                "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your",
                "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she",
                "her", "hers", "herself", "it", "its", "itself", "they", "them", "their",
                "theirs", "themselves", "what", "which", "who", "whom", "this", "that",
                "these", "those", "am", "is", "are", "was", "were", "be", "been", "being",
                "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an",
                "the", "and", "but", "if", "or", "because", "as", "until", "while", "of",
                "at", "by", "for", "with", "about", "against", "between", "into", "through",
                "during", "before", "after", "above", "below", "to", "from", "up", "down",
                "in", "out", "on", "off", "over", "under", "again", "further", "then", "once"
            }

        if preserve_negations:
            self.stop_words = raw_stops - self.CRITICAL_NEGATIONS
        else:
            self.stop_words = raw_stops

    @staticmethod
    def _clean_surface_text(text: str) -> str:
        """
        Cleans URLs, HTML tags, user mentions (@user), and non-alphabetical noise.
        """
        if not isinstance(text, str):
            text = str(text or "")

        # Lowercase
        text = text.lower()

        # Remove URLs
        text = re.sub(r"https?://\S+|www\.\S+", " ", text)

        # Remove HTML tags
        text = re.sub(r"<.*?>", " ", text)

        # Remove Twitter/social mentions (@user)
        text = re.sub(r"@\w+", " ", text)

        # Retain letters and basic apostrophes for contractions (e.g. don't -> don't)
        text = re.sub(r"[^a-z\s']", " ", text)

        # Remove isolated apostrophes and excessive whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @staticmethod
    def _get_wordnet_pos(treebank_tag: str):
        """
        Map Penn Treebank POS tag to WordNet POS character.
        Default to NOUN if unrecognized.
        """
        if treebank_tag.startswith("J"):
            return wordnet.ADJ
        elif treebank_tag.startswith("V"):
            return wordnet.VERB
        elif treebank_tag.startswith("N"):
            return wordnet.NOUN
        elif treebank_tag.startswith("R"):
            return wordnet.ADV
        else:
            return wordnet.NOUN

    def preprocess_text(self, text: str) -> str:
        """
        Full preprocessing pipeline:
        1. Clean surface noise (URLs, mentions, punctuation)
        2. Tokenize text into words
        3. Remove non-sentiment stop-words
        4. POS-tag tokens and apply WordNet lemmatization
        5. Return cleaned, lemmatized string
        """
        cleaned = self._clean_surface_text(text)
        if not cleaned:
            return ""

        # Tokenization (fallback to simple split if punkt tokenizer has issue)
        try:
            tokens = word_tokenize(cleaned)
        except Exception:
            tokens = cleaned.split()

        # Filter stop-words and short non-informative characters
        filtered_tokens = [
            token for token in tokens
            if token not in self.stop_words and len(token) > 1
        ]

        if not filtered_tokens:
            return ""

        # POS-Tagging for accurate lemmatization
        try:
            tagged = pos_tag(filtered_tokens)
            lemmatized_tokens = [
                self.lemmatizer.lemmatize(token, self._get_wordnet_pos(tag))
                for token, tag in tagged
            ]
        except Exception:
            # Fallback without POS tag if tagger resource is missing
            lemmatized_tokens = [
                self.lemmatizer.lemmatize(token) for token in filtered_tokens
            ]

        return " ".join(lemmatized_tokens)

    def preprocess_corpus(self, texts: List[str]) -> List[str]:
        """
        Preprocess an entire list or Series of texts.
        """
        return [self.preprocess_text(t) for t in texts]


if __name__ == "__main__":
    # Test script standalone
    preprocessor = TextPreprocessor(preserve_negations=True)
    sample = "The customer service wasn't helpful at all! Check https://status.example.com @support #fail"
    print("Original:", sample)
    print("Cleaned & Lemmatized:", preprocessor.preprocess_text(sample))
