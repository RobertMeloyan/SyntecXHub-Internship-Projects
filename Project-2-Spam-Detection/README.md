# 📩 SMS Spam Classification Pipeline

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikitlearn&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-Data%20Analysis-150458?logo=pandas&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-NLP-yellowgreen)
![Seaborn](https://img.shields.io/badge/Seaborn-Visualization-4c72b0)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

An end-to-end **Natural Language Processing (NLP)** pipeline that classifies SMS messages as **Spam** or **Ham** (legitimate), combining classic text preprocessing, exploratory data analysis, and a modular Scikit-Learn machine learning pipeline.

---

## 📋 Table of Contents

- [Overview & Features](#-overview--features)
- [Repository Structure](#-repository-structure)
- [Installation & Usage](#-installation--usage)
- [EDA & Feature Analysis Summary](#-eda--feature-analysis-summary)
- [Future Improvements](#-future-improvements)
- [License](#-license)

---

## 🔍 Overview & Features

Spam messages are a persistent problem for SMS and email platforms. This project builds a reliable text classification system that automatically flags spam messages using statistical NLP techniques rather than hand-written rules.

**Key Features:**

- 🧹 **Data Cleaning & Preprocessing** — handles null values, drops unnecessary index artifacts (e.g. `Unnamed: 0`), and normalizes raw text (lowercasing, removing headers/punctuation, whitespace cleanup).
- 📊 **Exploratory Data Analysis (EDA)** — visualizes the Spam vs. Ham class distribution with annotated Seaborn countplots to reveal class imbalance.
- 🧮 **Feature Engineering** — derives `char_length` and `word_count` statistics per message and compares their distributions across spam and ham classes.
- ⚙️ **Modular ML Pipeline** — bundles TF-IDF vectorization with classifiers (Naive Bayes / Logistic Regression) inside a single Scikit-Learn `Pipeline`, enabling clean training, evaluation, and reusable inference.
- 💾 **Model Persistence** — serializes the trained pipeline with `joblib` for reuse without retraining.

---

## 📁 Repository Structure

```
.
├── spam_detection.ipynb            # Main notebook: EDA, preprocessing, training & evaluation
├── spam_ham_dataset.csv            # Labeled SMS/email dataset (spam vs. ham)
├── spam_classifier_pipeline.joblib # Serialized trained pipeline (generated after running the notebook)
├── requirements.txt                # Project dependencies
└── README.md                       # Project documentation
```

---

## 🚀 Installation & Usage

1. **Clone the repository**
   ```bash
   git clone https://github.com/<your-username>/<your-repo-name>.git
   cd <your-repo-name>
   ```

2. **Create and activate a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate      # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install pandas numpy scikit-learn seaborn matplotlib nltk jupyter joblib
   ```

   Or, if a `requirements.txt` is provided:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download NLTK resources** (if the notebook uses NLTK tokenizers/stopwords)
   ```python
   import nltk
   nltk.download("stopwords")
   nltk.download("punkt")
   ```

5. **Launch the notebook**
   ```bash
   jupyter notebook spam_detection.ipynb
   ```

6. **Run all cells** to reproduce the EDA, train the models, and generate the serialized pipeline.

> ⚠️ Make sure `spam_ham_dataset.csv` is placed in the same directory as the notebook before running.

---

## 📈 EDA & Feature Analysis Summary

- **Class Distribution:** A countplot of Spam vs. Ham messages (with percentage annotations) shows the dataset's class balance — an important check before choosing evaluation metrics, since imbalanced classes can inflate accuracy scores.
- **Character Length & Word Count:** Two engineered features — `char_length` and `word_count` — are computed per message and visualized (via Seaborn histograms/KDE plots) split by class. Spam messages typically show distinct length patterns compared to ham, offering useful signal even before vectorization.
- **Text Preprocessing:** All messages are lowercased, stripped of header artifacts (e.g. `subject:`), cleaned of non-alphabetic characters, and whitespace-normalized before being fed into `TfidfVectorizer`.
- **Modeling:** TF-IDF features (`max_features=5000`, English stop words removed) are passed into both `MultinomialNB` and `LogisticRegression` pipelines, evaluated via `classification_report` and confusion matrices on a stratified 80/20 train-test split.

---

## 🔮 Future Improvements

- Incorporate additional NLP techniques such as **lemmatization/stemming** and **n-gram features** to capture more context.
- Experiment with more advanced models (e.g. **SVM**, **XGBoost**, or transformer-based embeddings like **BERT**) for potentially higher accuracy.
- Add **cross-validation** and **hyperparameter tuning** (e.g. `GridSearchCV`) for more robust model selection.
- Address class imbalance explicitly using techniques like **SMOTE** or class-weighting.
- Deploy the trained pipeline as a lightweight **API (Flask/FastAPI)** or **Streamlit app** for real-time spam checking.
- Expand evaluation with **ROC curves** and **precision-recall curves** for deeper threshold analysis.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

<p align="center">Made with ❤️ using <code>scikit-learn</code> and <code>NLTK</code></p>
