<div align="center">

# 💳 Credit Card Fraud Detection

### A production-ready fraud detection engine built on severely imbalanced transactional data — combining SMOTE resampling, XGBoost, and business-driven threshold optimization.

**SyntecXHub Virtual Internship — Project 3**

<br/>

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-EC4E20?style=for-the-badge)](https://xgboost.readthedocs.io/)
[![imbalanced-learn](https://img.shields.io/badge/imbalanced--learn-SMOTE-6A1B9A?style=for-the-badge)](https://imbalanced-learn.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

</div>

---

## 📑 Table of Contents

- [Project Overview](#-project-overview)
- [Dataset Description](#-dataset-description)
- [Project Workflow](#-project-workflow)
- [Data Leakage Investigation](#-data-leakage-investigation-the-key-finding)
- [Model Training & Evaluation](#-model-training--evaluation)
- [Business Decision Thresholds & Trade-offs](#-business-decision-thresholds--trade-offs)
- [Inference API](#-inference-api)
- [Technologies Used](#-technologies-used)
- [Installation & Usage](#-installation--usage)
- [Project Structure](#-project-structure)
- [Key Takeaways](#-key-takeaways)
- [License & Acknowledgments](#-license--acknowledgments)

---

## 🎯 Project Overview

Financial fraud detection is a textbook **extreme class imbalance** problem: fraudulent transactions make up roughly **0.3%** of all records. A model that naively predicts "not fraud" for every transaction would score 99.7% accuracy while catching exactly zero criminals — which is why this project never uses accuracy as a metric.

This repository implements an end-to-end pipeline that:

1. Explores the transaction data and isolates the transaction types where fraud actually occurs.
2. Engineers balance-error features — then **deliberately removes them** after an ablation study exposes them as a data-leakage source.
3. Applies **SMOTE** to the training set only, preserving test-set integrity.
4. Trains and compares **Random Forest** and **XGBoost** classifiers.
5. Tunes the decision threshold against the **Precision-Recall curve** to align model behavior with real business costs.
6. Ships a reusable `predict_fraud()` inference function returning an actionable `APPROVE` / `BLOCK / REVIEW` decision.

> **Headline result:** 93.23% precision, 88.01% recall, and a 0.9594 PR-AUC — catching 1,446 of 1,643 real frauds while blocking only 105 legitimate customers.

---

## 📊 Dataset Description

The dataset is a **mobile money transaction log** (PaySim-style) where each row represents a single financial transaction and its resulting account balances.

### Feature Breakdown

| Feature | Type | Description |
|---|---|---|
| `step` | Integer | Time unit of the simulation — 1 step = 1 hour of real-world time |
| `type` | Categorical | Transaction category: `PAYMENT`, `TRANSFER`, `CASH_OUT`, `CASH_IN`, `DEBIT` |
| `amount` | Float | Transaction value in local currency |
| `nameOrig` | String | Originating (sending) customer ID — **dropped** (identifier, no predictive signal) |
| `oldbalanceOrg` | Float | Sender's balance **before** the transaction |
| `newbalanceOrig` | Float | Sender's balance **after** the transaction |
| `nameDest` | String | Recipient customer/merchant ID — **dropped** (identifier) |
| `oldbalanceDest` | Float | Recipient's balance **before** the transaction |
| `newbalanceDest` | Float | Recipient's balance **after** the transaction |
| `isFraud` | Binary | 🎯 **Target variable** — 1 if the transaction was fraudulent |
| `isFlaggedFraud` | Binary | The legacy rule-based system's flag — **dropped** (it's a competing model's output, not a genuine input feature) |

### Class Distribution

| Class | Share | Note |
|---|---|---|
| **Legitimate (0)** | ~99.7% | Overwhelming majority |
| **Fraud (1)** | **~0.3%** | Severe imbalance — drives every modeling decision in this project |

### 🔍 Critical EDA Insight: Fraud Only Lives in Two Transaction Types

Grouping fraud counts by `type` revealed that **100% of fraudulent transactions occur in only `TRANSFER` and `CASH_OUT`**. `PAYMENT`, `CASH_IN`, and `DEBIT` transactions contain zero fraud cases.

The dataset was therefore filtered to these two types — a domain-driven reduction that removes millions of irrelevant rows, dramatically speeds up training, and improves the effective signal-to-noise ratio without discarding a single fraud case.

---

## 🔬 Project Workflow

### 1️⃣ Exploratory Data Analysis (EDA)
- Loaded the dataset and inspected structure, dtypes, and shape.
- Verified **zero missing values** and checked for duplicate records.
- Visualized the `isFraud` class distribution (count plot) to quantify the imbalance.
- Cross-tabulated fraud rate **by transaction type** using a log-scaled count plot — the analysis that surfaced the `TRANSFER`/`CASH_OUT` insight above.

### 2️⃣ Feature Engineering — Balance Error Features
Two "accounting consistency" features were engineered to capture whether the reported balances actually reconcile with the transaction amount:

```python
errorBalanceOrig = newbalanceOrig + amount - oldbalanceOrg
errorBalanceDest = oldbalanceDest + amount - newbalanceDest
```

The intuition: in a legitimate transaction the books should balance, so a non-zero error signals something anomalous. **These features turned out to be too good — see the leakage section below.**

### 3️⃣ Data Preprocessing
- Dropped identifier columns (`nameOrig`, `nameDest`) and the competing rule-engine output (`isFlaggedFraud`).
- One-hot encoded `type` with `drop_first=True` to avoid the dummy variable trap.
- **Stratified** 80/20 train-test split (`random_state=42`) to preserve the fraud ratio in both splits.

### 4️⃣ Handling Class Imbalance — SMOTE
**SMOTE** (Synthetic Minority Over-sampling Technique) was applied with `sampling_strategy=0.1`, synthesizing minority-class examples until fraud represents 10% of the training set.

> [!IMPORTANT]
> SMOTE is fitted on the **training set only**, *after* the train-test split. Resampling before splitting would leak synthetic points derived from test-set neighbors into training — inflating scores and producing a model that fails in production. The 0.1 target ratio is also a deliberate choice over full 1:1 balancing, which would over-distort the feature space and hurt precision.

---

## 🚨 Data Leakage Investigation (The Key Finding)

The first Random Forest — trained **with** `errorBalanceOrig` and `errorBalanceDest` — returned near-perfect precision and recall. In fraud detection, a perfect score is a red flag, not a victory.

An **ablation study** confirmed the suspicion: the engineered error-balance features encode the outcome almost deterministically, because of how the simulated balances are generated for fraudulent records. The model wasn't learning fraud patterns — it was reading the answer key.

| Model Variant | Features | Outcome |
|---|---|---|
| ❌ Leaky model | With `errorBalance*` | ~100% precision/recall — **unrealistic, discarded** |
| ✅ Clean model | Without `errorBalance*` | Realistic performance, **production-valid** |

**All final results below come from the clean model**, trained strictly on genuine inputs: `step`, `amount`, `oldbalanceOrg`, `newbalanceOrig`, `oldbalanceDest`, `newbalanceDest`, and the one-hot encoded `type`.

> [!NOTE]
> Choosing the honest 88% recall model over the flattering 100% one is the single most important engineering decision in this project. Leakage that survives into production doesn't just degrade — it collapses.

---

## 🤖 Model Training & Evaluation

Two tree-based ensembles were trained on the SMOTE-resampled, leakage-free training set:

<table>
<thead>
<tr><th>Model</th><th>Configuration</th><th>Role</th></tr>
</thead>
<tbody>
<tr>
<td><b>Random Forest</b></td>
<td><code>n_estimators=100</code>, <code>random_state=42</code>, <code>n_jobs=-1</code></td>
<td>Baseline + feature-importance analysis</td>
</tr>
<tr>
<td><b>XGBoost</b> ⭐</td>
<td><code>n_estimators=100</code>, <code>learning_rate=0.1</code>, <code>max_depth=6</code>, <code>random_state=42</code>, <code>n_jobs=-1</code></td>
<td><b>Final production model</b></td>
</tr>
</tbody>
</table>

### Evaluation Metrics Used

| Metric | Why It's Used Here |
|---|---|
| **Precision** | Of the transactions we block, how many were actually fraud? Directly measures customer friction. |
| **Recall** | Of all real frauds, how many did we catch? Directly measures financial loss prevented. |
| **F1-Score** | Harmonic mean — the balance point between the two competing costs above. |
| **PR-AUC** (Average Precision) | **The headline metric.** On data this imbalanced, ROC-AUC is misleadingly optimistic because the huge negative class dominates the false-positive rate. PR-AUC focuses squarely on minority-class performance. |
| **Confusion Matrix** | Translates percentages into the counts a business stakeholder actually reasons about. |

### 🏆 Final Performance (XGBoost @ Optimized Threshold)

<table>
<thead>
<tr><th>Metric</th><th>Score</th><th>Business Interpretation</th></tr>
</thead>
<tbody>
<tr><td><b>Precision</b></td><td><b>93.23%</b></td><td>Over 9 in 10 blocked transactions are genuinely fraudulent</td></tr>
<tr><td><b>Recall</b></td><td><b>88.01%</b></td><td>Caught <b>1,446</b> of <b>1,643</b> actual frauds in the test set</td></tr>
<tr><td><b>F1-Score</b></td><td><b>0.9054</b></td><td>Strong balance between catching fraud and avoiding false alarms</td></tr>
<tr><td><b>PR-AUC</b></td><td><b>0.9594</b></td><td>Excellent ranking capability across all possible thresholds</td></tr>
<tr><td><b>False Positive Rate</b></td><td><b>~0.02%</b></td><td>Only 105 legitimate customers inconvenienced</td></tr>
</tbody>
</table>

---

## ⚖️ Business Decision Thresholds & Trade-offs

A classifier's default 0.50 cutoff is an arbitrary statistical convention, not a business decision. The real question is: **what does each type of mistake cost us?**

| Error Type | What Happens | Business Cost |
|---|---|---|
| **False Negative** (missed fraud) | Fraudulent transaction approved | Direct financial loss + chargeback fees + liability |
| **False Positive** (false alarm) | Legitimate customer blocked | Customer frustration, support ticket, churn risk, lost revenue |

### Threshold Optimization

The optimal threshold was derived by sweeping the **Precision-Recall curve** and selecting the point that maximizes F1:

```python
precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_proba)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
best_threshold = thresholds[np.argmax(f1_scores)]   # → 0.8465
```

### 📈 Impact of Moving from 0.50 → 0.8465

<table>
<thead>
<tr><th>Metric</th><th>Default (0.50)</th><th>Optimized (0.8465)</th><th>Change</th></tr>
</thead>
<tbody>
<tr><td><b>Precision</b></td><td>57.50%</td><td><b>93.23%</b></td><td>🟢 +35.7 pts</td></tr>
<tr><td><b>False Positives</b></td><td>1,176</td><td><b>105</b></td><td>🟢 −91% customer friction</td></tr>
<tr><td><b>Recall</b></td><td>Slightly higher</td><td>88.01%</td><td>🟡 Small, deliberate trade</td></tr>
</tbody>
</table>

By raising the confidence bar, the system eliminates **over 1,000 unnecessary customer blocks** while still catching ~88% of fraud. That's a trade most payment businesses would take immediately.

### Tuning the Threshold for Your Risk Appetite

| Business Priority | Threshold Direction | Resulting Behavior |
|---|---|---|
| 🛡️ **Maximum security** (high-value transfers, regulated accounts) | **Lower** the threshold | Higher recall, more false alarms — catch everything, accept friction |
| 😊 **Maximum customer experience** (low-value, high-volume retail) | **Raise** the threshold | Higher precision, fewer blocks — only stop near-certain fraud |
| ⚖️ **Balanced** (this project's default) | **0.8465** (F1-optimal) | Best overall equilibrium of the two costs |

> [!TIP]
> The threshold is persisted **inside the model artifact**, so production inference automatically applies the tuned cutoff rather than silently defaulting back to 0.50.

---

## 🔌 Inference API

The trained model, its optimal threshold, and the exact feature ordering are bundled into a single artifact via `joblib`, and exposed through a simple prediction function:

```python
sample_transaction = {
    'step': 1,
    'amount': 181.0,
    'oldbalanceOrg': 181.0,
    'newbalanceOrig': 0.0,
    'oldbalanceDest': 0.0,
    'newbalanceDest': 0.0,
    'type_TRANSFER': True
}

result = predict_fraud(sample_transaction)
# → {'is_fraud': True, 'fraud_probability': 0.97, 'action': 'BLOCK / REVIEW'}
```

**Design notes:**
- The artifact stores `{'model', 'threshold', 'features'}` together, so the deployed model can never drift out of sync with its tuned threshold or expected feature schema.
- Incoming transactions are `reindex`-ed against the saved feature list (`fill_value=0`), making the function resilient to missing or reordered one-hot columns.
- The output includes a human-readable `action` field, so the function slots directly into a payment-gateway decision flow.

---

## 🛠️ Technologies Used

| Category | Tools |
|---|---|
| **Language** | Python 3.10+ |
| **Data Manipulation** | `pandas`, `numpy` |
| **Visualization** | `matplotlib`, `seaborn` |
| **Machine Learning** | `scikit-learn` (Random Forest, metrics, splitting), `xgboost` |
| **Imbalance Handling** | `imbalanced-learn` (SMOTE) |
| **Model Persistence** | `joblib` |
| **Environment** | Jupyter Notebook |

---

## ⚙️ Installation & Usage

### Prerequisites
- Python **3.10+**
- `pip` or `conda`
- The dataset CSV placed in the project root (see note below)

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/credit-card-fraud-detection.git
cd credit-card-fraud-detection
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate      # On Windows: .venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

<details>
<summary><code>requirements.txt</code></summary>

```
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
xgboost>=2.0.0
imbalanced-learn>=0.11.0
joblib>=1.3.0
jupyter>=1.0.0
```

</details>

### 4. Add the Dataset

Place the transaction dataset in the project root as **`AIML Dataset.csv`** (or update the path in the data-loading cell of the notebook).

### 5. Run the Notebook

```bash
jupyter notebook Financial_Fraud_Detection.ipynb
```

Execute cells top-to-bottom (`Cell → Run All`). The pipeline will train both models, run the leakage ablation, optimize the threshold, and export `fraud_detection_xgb.pkl`.

### 6. Use the Saved Model

```python
import joblib
from your_module import predict_fraud

result = predict_fraud({
    'step': 5,
    'amount': 250000.0,
    'oldbalanceOrg': 250000.0,
    'newbalanceOrig': 0.0,
    'oldbalanceDest': 0.0,
    'newbalanceDest': 0.0,
    'type_TRANSFER': True
})
print(result['action'])
```

> [!WARNING]
> `joblib` artifacts are pickle-based and execute arbitrary code on load. Only load `.pkl` files you produced yourself or obtained from a trusted source.

---

## 📁 Project Structure

```
credit-card-fraud-detection/
├── README.md                          # You are here
├── Financial_Fraud_Detection.ipynb    # Main notebook: full EDA → training → threshold tuning
├── requirements.txt                   # Python dependencies
├── AIML Dataset.csv                   # Transaction dataset (not tracked in git)
├── fraud_detection_xgb.pkl            # Exported model + threshold + feature schema
└── LICENSE                            # MIT License
```

---

## 💡 Key Takeaways

- **Accuracy is a trap on imbalanced data.** A 99.7%-accurate model here can be completely useless. PR-AUC, precision, and recall are the metrics that matter.
- **Domain-driven filtering beats brute force.** Recognizing that fraud only occurs in `TRANSFER` and `CASH_OUT` cut the problem size massively at zero cost to fraud coverage.
- **Suspiciously perfect scores mean leakage, not success.** The ablation study that removed `errorBalanceOrig`/`errorBalanceDest` was the difference between a demo and a deployable model.
- **Resample only the training set.** SMOTE before the split is one of the most common — and most silently damaging — mistakes in imbalanced classification.
- **The default 0.50 threshold is a placeholder, not a decision.** Tuning it to 0.8465 cut false positives by 91% and raised precision by nearly 36 points.
- **Ship the threshold with the model.** Persisting the cutoff alongside the weights prevents a whole class of production drift bugs.

---

## 🗺️ Future Improvements

- [ ] Add cost-sensitive threshold selection using actual monetary values (avg. fraud loss vs. avg. cost of a false block) instead of F1.
- [ ] Benchmark against `LightGBM` and a `CatBoost` baseline.
- [ ] Add SHAP explanations so analysts can see *why* a transaction was flagged.
- [ ] Introduce time-aware validation (split by `step`) to simulate true production conditions where the model predicts on future transactions.
- [ ] Engineer velocity features (transactions per account per time window) as non-leaky behavioral signals.
- [ ] Wrap `predict_fraud()` in a FastAPI endpoint with input validation and request logging.
- [ ] Add drift monitoring for feature distributions and fraud rate over time.

---

## 📄 License & Acknowledgments

**License:** Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for details.

**Acknowledgments:**
- **SyntecXHub Virtual Internship** — project brief and mentorship (Project 3).
- **Dataset:** Mobile money transaction simulation data (PaySim-style synthetic financial dataset).
- **Libraries:** [scikit-learn](https://scikit-learn.org/), [XGBoost](https://xgboost.readthedocs.io/), and [imbalanced-learn](https://imbalanced-learn.org/).

---

<div align="center">

If this project helped you, consider leaving a ⭐

</div>
