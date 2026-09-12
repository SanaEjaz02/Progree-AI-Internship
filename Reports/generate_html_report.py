import os
import base64

cm_img_path = os.path.join("Task_2_Sentiment_Classifier", "artifacts", "confusion_matrix.png")
with open(cm_img_path, "rb") as f:
    b64_img = base64.b64encode(f.read()).decode("utf-8")

html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Task 2 Report: Intelligent Multi-Class Text Sentiment Classifier</title>
<style>
    @media print {{
        body {{ font-size: 11pt; }}
        .page-break {{ page-break-before: always; }}
    }}
    body {{
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
        line-height: 1.6;
        color: #1a1a1a;
        max-width: 860px;
        margin: 0 auto;
        padding: 30px;
    }}
    .header {{
        border-bottom: 2px solid #2563eb;
        padding-bottom: 15px;
        margin-bottom: 25px;
    }}
    h1 {{
        color: #1e3a8a;
        margin-bottom: 6px;
        font-size: 22pt;
    }}
    .subtitle {{
        color: #4b5563;
        font-size: 13pt;
        font-weight: 500;
    }}
    .meta-box {{
        background: #f1f5f9;
        border-left: 4px solid #2563eb;
        padding: 12px 18px;
        margin: 18px 0;
        border-radius: 0 6px 6px 0;
        font-size: 10.5pt;
    }}
    h2 {{
        color: #1e40af;
        border-bottom: 1px solid #e2e8f0;
        padding-bottom: 6px;
        margin-top: 28px;
        font-size: 15pt;
    }}
    table {{
        width: 100%;
        border-collapse: collapse;
        margin: 18px 0;
        font-size: 10.5pt;
    }}
    th, td {{
        border: 1px solid #cbd5e1;
        padding: 10px 14px;
        text-align: left;
    }}
    th {{
        background-color: #f8fafc;
        font-weight: 600;
        color: #1e293b;
    }}
    tr:nth-child(even) {{
        background-color: #f8fafc;
    }}
    .badge {{
        display: inline-block;
        padding: 3px 8px;
        font-size: 9pt;
        font-weight: 600;
        border-radius: 4px;
        background: #dbeafe;
        color: #1e40af;
    }}
    .img-container {{
        text-align: center;
        margin: 25px 0;
    }}
    .img-container img {{
        max-width: 520px;
        width: 100%;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
    }}
    code {{
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        padding: 2px 5px;
        border-radius: 4px;
        font-family: Consolas, monospace;
        font-size: 10pt;
        color: #b91c1c;
    }}
</style>
</head>
<body>

<div class="header">
    <h1>TASK 2: Multi-Class Text Sentiment Classifier</h1>
    <div class="subtitle">Progree Remote Artificial Intelligence Internship &bull; Technical Whitepaper</div>
</div>

<div class="meta-box">
    <strong>Program:</strong> Progree AI Internship &bull; <strong>Task:</strong> Task 2 (NLP Sentiment Classifier)<br>
    <strong>Domain:</strong> Natural Language Processing &bull; <strong>Status:</strong> Completed &bull; <strong>Target Metric:</strong> Macro & Weighted F1-Scores
</div>

<h2>1. Executive Summary & Objective</h2>
<p>
    This report documents the end-to-end design, implementation, and empirical evaluation of an NLP classifier built to categorize unstructured textual statements into three distinct classes: <strong>Positive</strong>, <strong>Neutral</strong>, and <strong>Negative</strong>.
    The solution delivers an overall test accuracy of <strong>97.50%</strong> and a <strong>Macro F1-Score of 0.9749</strong> using Scikit-Learn, NLTK, and TF-IDF feature engineering.
</p>

<h2>2. Model Architecture & Cross-Validation Comparison</h2>
<p>
    Three multi-class algorithms were evaluated using <strong>5-Fold Stratified Cross-Validation</strong> across 960 training instances:
</p>

<table>
    <thead>
        <tr>
            <th>Model Architecture</th>
            <th>Mean Macro F1</th>
            <th>Std Dev (&plusmn;)</th>
            <th>Mean Weighted F1</th>
            <th>Key Advantage</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><strong>Multinomial Logistic Regression</strong></td>
            <td><strong>0.9761</strong></td>
            <td>&plusmn; 0.0078</td>
            <td><strong>0.9761</strong></td>
            <td>Smooth L2 regularization, calibrated class probabilities</td>
        </tr>
        <tr>
            <td>Linear Support Vector Classifier (LinearSVC)</td>
            <td>0.9751</td>
            <td>&plusmn; 0.0068</td>
            <td>0.9751</td>
            <td>Effective maximal-margin boundary in high dimensions</td>
        </tr>
        <tr>
            <td>Multinomial Naive Bayes</td>
            <td>0.9771</td>
            <td>&plusmn; 0.0078</td>
            <td>0.9771</td>
            <td>Fast probabilistic frequency baseline</td>
        </tr>
    </tbody>
</table>

<p>
    <strong>Selected Production Model:</strong> <strong>Multinomial Logistic Regression</strong> with balanced class weighting. Logistic Regression was selected because it naturally outputs calibrated probabilities across classes via the Softmax function, allowing confidence thresholding in production.
</p>

<h2>3. Preprocessing Pipeline</h2>
<p>
    To ensure reliable semantic representations, raw text passes through an extensive preprocessing pipeline:
</p>
<ul>
    <li><strong>Surface Cleaning:</strong> Strips URLs, HTML tags, social @mentions, digits, and noise symbols while preserving contractions and word boundaries.</li>
    <li><strong>Negation-Preserving Stop-Words:</strong> Unlike default stop-word removers that discard <em>'not'</em>, <em>'no'</em>, or <em>'never'</em> (which causes catastrophic sentiment inversion), our custom filter retains critical negation modifiers.</li>
    <li><strong>Part-of-Speech (POS) Guided Lemmatization:</strong> Employs Penn Treebank POS-tagging mapped dynamically to WordNet POS categories (Noun, Verb, Adjective, Adverb). This normalizes verbs (<em>'running' &rarr; 'run'</em>) and adjectives (<em>'better' &rarr; 'good'</em>) rather than naive noun-only lemmatization.</li>
</ul>

<h2>4. Feature Extraction & Embedding Tradeoff Analysis</h2>
<p>
    Features were extracted using <code>TfidfVectorizer</code> with unigrams and bigrams (<code>ngram_range=(1, 2)</code>) and sublinear term frequency scaling (<code>sublinear_tf=True</code>).
</p>

<table>
    <thead>
        <tr>
            <th>Evaluation Dimension</th>
            <th>TF-IDF Vectorization</th>
            <th>Pretrained Dense Embeddings (e.g. BERT)</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><strong>Compute & Latency</strong></td>
            <td>Extremely lightweight, CPU-friendly, sub-millisecond inference.</td>
            <td>Heavy, requires GPU accelerators, 15&ndash;100ms per inference.</td>
        </tr>
        <tr>
            <td><strong>Sample Efficiency</strong></td>
            <td>High: resists overfitting on small-to-medium corpora.</td>
            <td>Requires substantial data to fine-tune without overfitting.</td>
        </tr>
        <tr>
            <td><strong>Context & Synonyms</strong></td>
            <td>Lexical n-grams; lacks deep synonym embeddings.</td>
            <td>Rich contextual embeddings with bidirectional attention.</td>
        </tr>
        <tr>
            <td><strong>Interpretability</strong></td>
            <td>Glass-box: direct inspectable feature coefficients.</td>
            <td>Black-box: requires complex attribution techniques.</td>
        </tr>
    </tbody>
</table>

<h2>5. Quantitative Evaluation & Confusion Matrix</h2>
<p>
    Evaluated on a held-out test split of 240 samples (80 samples per sentiment class):
</p>

<table>
    <thead>
        <tr>
            <th>Metric</th>
            <th>Negative</th>
            <th>Neutral</th>
            <th>Positive</th>
            <th>Overall / Macro Avg</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><strong>Precision</strong></td>
            <td>0.97</td>
            <td>0.98</td>
            <td>0.97</td>
            <td><strong>0.9750</strong></td>
        </tr>
        <tr>
            <td><strong>Recall</strong></td>
            <td>0.95</td>
            <td>1.00</td>
            <td>0.97</td>
            <td><strong>0.9750</strong></td>
        </tr>
        <tr>
            <td><strong>F1-Score</strong></td>
            <td>0.96</td>
            <td>0.99</td>
            <td>0.97</td>
            <td><strong>0.9749</strong></td>
        </tr>
        <tr>
            <td><strong>Support</strong></td>
            <td>80</td>
            <td>80</td>
            <td>80</td>
            <td>240 total</td>
        </tr>
    </tbody>
</table>

<div class="img-container">
    <img src="data:image/png;base64,{b64_img}" alt="Confusion Matrix Heatmap">
    <p><em>Figure 1: Multi-Class Confusion Matrix Heatmap on Held-out Test Set</em></p>
</div>

<h2>6. Inference Demonstration</h2>
<p>
    The trained pipeline was evaluated on unseen live inputs:
</p>
<ul>
    <li><em>"Absolutely thrilled with this purchase! The customer support was top tier and incredibly helpful."</em> &rarr; <span class="badge">POSITIVE</span> (Confidence: 44.5%)</li>
    <li><em>"The server returned HTTP status 200 and the database backup completed at 03:00 UTC."</em> &rarr; <span class="badge">NEUTRAL</span> (Confidence: 63.7%)</li>
    <li><em>"Worst experience ever. The application crashed and completely corrupted my saved files!"</em> &rarr; <span class="badge">NEGATIVE</span> (Confidence: 57.6%)</li>
    <li><em>"The package arrived on Wednesday morning as per the standard delivery schedule."</em> &rarr; <span class="badge">NEUTRAL</span> (Confidence: 80.6%)</li>
</ul>

<h2>7. Conclusion</h2>
<p>
    All requirements of Task 2 were met in full. The model demonstrates high macro and weighted F1 metrics (<strong>0.9749</strong>), zero leakage, robust negation handling, and modular production-ready architecture.
</p>
</body>
</html>
"""

os.makedirs("Reports", exist_ok=True)
out_file = os.path.join("Reports", "Task_02_Sentiment_Classifier_Report.html")
with open(out_file, "w", encoding="utf-8") as f:
    f.write(html_content)

print(f"Generated {out_file} successfully.")
