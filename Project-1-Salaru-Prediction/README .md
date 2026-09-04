# 🛒 Ecommerce Customer Spending Prediction

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikitlearn&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-Data%20Analysis-150458?logo=pandas&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

A **Multiple Linear Regression** model that predicts the **Yearly Amount Spent** by ecommerce customers based on their engagement with a company's website, mobile app, and membership tenure.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Dataset](#-dataset)
- [Technologies Used](#-technologies-used)
- [Installation & How to Run](#-installation--how-to-run)
- [Model Evaluation & Results](#-model-evaluation--results)
- [Key Takeaways / Business Insights](#-key-takeaways--business-insights)
- [Project Structure](#-project-structure)
- [License](#-license)

---

## 🔍 Overview

An ecommerce company sells clothing online but also offers styling advice via in-person sessions. Customers can access the service through the **website** or the **mobile app**. This project builds a regression model to understand which of these engagement channels — along with membership length — most strongly drives **yearly customer spending**, helping the business decide where to focus investment: the app or the website.

---

## 📊 Dataset

The model is trained on `Ecommerce Customers.csv`, which contains the following features:

| Feature | Description |
|---|---|
| `Avg. Session Length` | Average length of in-store style advice sessions (minutes) |
| `Time on App` | Average time spent on the mobile app (minutes) |
| `Time on Website` | Average time spent on the website (minutes) |
| `Length of Membership` | Number of years the customer has been a member |
| `Yearly Amount Spent` | **Target variable** — total amount spent per year ($) |

---

## 🛠 Technologies Used

- **Python 3.10+**
- **pandas** — data loading and manipulation
- **scikit-learn** — model training, train/test split, and evaluation metrics

---

## 🚀 Installation & How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/<your-username>/<your-repo-name>.git
   cd <your-repo-name>
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate      # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install pandas scikit-learn jupyter
   ```

4. **Launch the notebook**
   ```bash
   jupyter notebook linear_regression.ipynb
   ```

5. **Run all cells** to load the data, train the model, and view the evaluation metrics.

> ⚠️ Make sure `Ecommerce Customers.csv` is in the same directory as the notebook before running.

---

## 📈 Model Evaluation & Results

The dataset was split **80% train / 20% test** (`test_size=0.2, random_state=42`), and a `LinearRegression` model was fit on the training set.

### Performance Metrics

| Metric | Value | Description |
|---|---|---|
| **MAE** (Mean Absolute Error) | `8.56` | Average absolute prediction error, in dollars |
| **MSE** (Mean Squared Error) | `109.86` | Average squared prediction error |
| **R² Score** | `0.9778` | ~97.78% of variance in spending explained by the model |

### Feature Coefficients

| Feature | Coefficient |
|---|---|
| Avg. Session Length | `25.60` |
| Time on App | `38.79` |
| Time on Website | `0.31` |
| Length of Membership | `61.90` |

> Each coefficient represents the expected change in **Yearly Amount Spent ($)** for a one-unit increase in that feature, holding all other features constant.

---

## 💡 Key Takeaways / Business Insights

- **Length of Membership has the strongest impact** (`+61.90`) — long-term, loyal customers spend significantly more per year. Retention strategies and loyalty programs are likely the highest-leverage investment.
- **Time on App (`+38.79`) far outweighs Time on Website (`+0.31`)** — customers who engage more with the **mobile app** spend substantially more, while extra time on the website barely moves the needle. This strongly suggests the company should **prioritize the mobile app experience** over the website.
- **Avg. Session Length (`+25.60`)** also has a meaningful positive effect — improving the quality of in-person/style-advice sessions could drive additional revenue.
- With an **R² of 0.9778**, the model explains almost all of the variance in customer spending, and with a low **MAE of $8.56**, predictions are, on average, very close to actual spending — making this a reliable model for forecasting and guiding strategic decisions.

**Bottom line:** 📱 Double down on the **app experience** and **membership retention** — these are the two biggest levers for increasing revenue per customer.

---

## 📁 Project Structure

```
.
├── linear_regression.ipynb     # Main notebook: EDA, model training & evaluation
├── Ecommerce Customers.csv     # Dataset (not included — add your own)
└── README.md                   # Project documentation
```

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

<p align="center">Made with ❤️ and <code>scikit-learn</code></p>
