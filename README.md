# Bankruptcy Prediction Machine Learning Project

## Project Overview

This project is a machine learning-based **Bankruptcy Prediction System** that predicts whether a company is likely to be **healthy** or **bankrupt** using financial ratio data.

The project follows the complete data science workflow, including data exploration, data cleaning, feature engineering, model building, model evaluation, hyperparameter tuning, and web app testing.

A simple **Streamlit web app** is also included to test predictions locally using a browser.

---

## Problem Statement

Company bankruptcy prediction is important for investors, banks, financial analysts, and business decision-makers.

The goal of this project is to build a machine learning model that can identify companies with possible bankruptcy risk based on financial ratio data.

This is a **binary classification problem**.

Target variable:

| Value | Meaning          |
| ----- | ---------------- |
| `0`   | Healthy company  |
| `1`   | Bankrupt company |

---

## Dataset Information

The dataset used in this project is:

```text
bankruptcy.csv
```

The dataset contains financial ratio data of companies.

Dataset summary:

| Description        |      Value |
| ------------------ | ---------: |
| Total records      |      6,819 |
| Original columns   |         95 |
| Financial features |         94 |
| Target column      | `Bankrupt` |

Target variable distribution:

| Class | Meaning  | Count | Percentage |
| ----- | -------- | ----: | ---------: |
| 0     | Healthy  | 6,599 |     96.77% |
| 1     | Bankrupt |   220 |      3.23% |

The dataset is highly imbalanced because the number of healthy companies is much higher than the number of bankrupt companies.

Because of this imbalance, accuracy alone is not enough to evaluate the machine learning models. Metrics such as precision, recall, F1-score, and ROC-AUC are also used.

---

## Project Structure

```text
bankruptcy-prediction-ml/
│
├── data/
│   ├── bankruptcy.csv
│   └── bankruptcy_cleaned.csv
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_data_cleaning.ipynb
│   └── 03_model_building.ipynb
│
├── src/
│   └── utils.py
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Files Description

| File / Folder                       | Description                                                    |
| ----------------------------------- | -------------------------------------------------------------- |
| `data/bankruptcy.csv`               | Original dataset                                               |
| `data/bankruptcy_cleaned.csv`       | Cleaned dataset created after preprocessing                    |
| `notebooks/01_eda.ipynb`            | Exploratory Data Analysis notebook                             |
| `notebooks/02_data_cleaning.ipynb`  | Data cleaning and preprocessing notebook                       |
| `notebooks/03_model_building.ipynb` | Model training, evaluation, tuning, and final results notebook |
| `src/utils.py`                      | Utility functions used in the project                          |
| `app.py`                            | Streamlit web app for testing predictions                      |
| `requirements.txt`                  | Python libraries required for the project                      |
| `README.md`                         | Project documentation                                          |
| `.gitignore`                        | Files and folders ignored by Git                               |

---

## Technologies Used

* Python
* Jupyter Notebook
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Streamlit
* Visual Studio Code
* GitHub Desktop

---

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/bankruptcy-prediction-ml.git
```

Move into the project folder:

```bash
cd bankruptcy-prediction-ml
```

---

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate the virtual environment on Windows:

```bash
.venv\Scripts\activate
```

---

### 3. Install Required Libraries

```bash
pip install -r requirements.txt
```

---

### 4. Add Jupyter Kernel

```bash
pip install ipykernel
```

```bash
python -m ipykernel install --user --name bankruptcy-ml --display-name "Python (Bankruptcy ML)"
```

---

## How to Run the Project

Run the notebooks in this order:

```text
1. notebooks/01_eda.ipynb
2. notebooks/02_data_cleaning.ipynb
3. notebooks/03_model_building.ipynb
```

The second notebook creates the cleaned dataset:

```text
data/bankruptcy_cleaned.csv
```

The third notebook uses the cleaned dataset for model building.

---

## Run Jupyter Notebook Locally

Start Jupyter Notebook using:

```bash
jupyter notebook
```

Then open:

```text
http://localhost:8888
```

Open the notebooks from the `notebooks` folder and run them in order.

---

## Exploratory Data Analysis

The EDA notebook includes:

* Dataset overview
* Data type checking
* Missing value checking
* Duplicate record checking
* Target variable distribution
* Class imbalance analysis
* Numerical feature analysis
* Outlier detection
* Bivariate analysis
* Correlation analysis
* Pair plot analysis
* Final EDA summary

### Key EDA Findings

* The dataset contains 6,819 records and 95 original columns.
* The target variable is `Bankrupt`.
* The dataset is highly imbalanced.
* Only 3.23% of companies are bankrupt.
* Profitability-related features are generally lower for bankrupt companies.
* Liability and debt-related features are generally higher for bankrupt companies.
* Several numerical features contain outliers.
* Some financial ratio features are highly correlated with each other.

---

## Data Cleaning and Preprocessing

The data cleaning notebook includes:

* Target variable conversion
* Missing value checking
* Invalid value checking
* Duplicate record checking
* Imputation verification
* Feature engineering
* Boolean and categorical feature handling
* Saving the cleaned dataset

### Feature Engineering

Three new aggregate financial features were created:

```text
AvgProfitability
AvgLeverage
AvgLiquidity
```

After feature engineering, the cleaned dataset contains:

```text
6819 rows × 98 columns
```

The cleaned dataset is saved as:

```text
data/bankruptcy_cleaned.csv
```

---

## Machine Learning Models

The following machine learning models were trained and evaluated:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. K-Nearest Neighbors
5. Support Vector Machine
6. Gradient Boosting
7. Naive Bayes
8. Tuned Random Forest

---

## Model Evaluation Metrics

The models were evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix
* ROC Curve
* ROC-AUC
* 5-Fold Cross-Validation

Because the dataset is highly imbalanced, the most important metrics are:

* Recall
* Precision
* F1-score

---

## Why Accuracy Alone Is Not Enough

The dataset is highly imbalanced.

Most companies are healthy, so a model can achieve high accuracy by predicting most companies as healthy.

However, this is not useful if the model fails to identify bankrupt companies.

For bankruptcy prediction, recall is very important because missing a bankrupt company can be costly.

---

## Model Comparison

Several models achieved high accuracy, but their ability to identify bankrupt companies was different.

Some models had higher precision, while others had higher recall.

For this project, the final model was selected by considering the balance between precision and recall.

---

## Final Model Result

The best final model was:

```text
Tuned Random Forest
```

Final performance:

| Metric    |  Score |
| --------- | -----: |
| Accuracy  | 0.9633 |
| Precision | 0.4559 |
| Recall    | 0.7045 |
| F1-score  | 0.5536 |

The Tuned Random Forest model was selected because it provided the best balance between precision and recall.

---

## Feature Importance

The tuned Random Forest model identified several important financial indicators, including:

* Persistent EPS
* Net income to total assets
* Borrowing dependency
* ROA ratios
* Retained earnings
* Debt ratio
* Net worth to assets
* Profitability indicators
* Liability-related indicators

These features show that bankruptcy prediction is influenced by:

* Profitability
* Debt level
* Asset strength
* Earnings strength
* Financial stability

---

## Streamlit Web App

This project includes a simple Streamlit web app for testing predictions locally.

The web app includes:

* Project overview
* Dataset preview
* Model performance
* Manual prediction form
* Bankruptcy probability output

---

## How to Run the Web App

Activate the virtual environment:

```bash
.venv\Scripts\activate
```

Run the Streamlit app:

```bash
streamlit run app.py
```

The app will open at:

```text
http://localhost:8501
```

---

## Web App Features

The Streamlit app allows users to:

* View basic project information
* View dataset summary
* Check model performance
* Enter financial values manually
* Predict whether a company is healthy or bankrupt
* View bankruptcy probability

Prediction output:

| Output               | Meaning                          |
| -------------------- | -------------------------------- |
| Healthy / Low Risk   | Company is predicted as healthy  |
| Bankrupt / High Risk | Company is predicted as bankrupt |

---

## Requirements

The project uses the following Python libraries:

```text
jupyter
pandas
numpy
matplotlib
seaborn
scikit-learn
streamlit
```

Install all requirements using:

```bash
pip install -r requirements.txt
```

---

## Suggested GitHub Commit History

```text
Add initial project files and folder structure
Complete exploratory data analysis notebook
Complete data cleaning and preprocessing notebook
Complete model building and evaluation notebook
Add Streamlit web app for bankruptcy prediction
Update README documentation
```

---

## Future Improvements

This project can be improved further by:

* Applying SMOTE or other imbalance handling techniques
* Trying advanced boosting models such as XGBoost or LightGBM
* Performing deeper hyperparameter tuning
* Applying threshold tuning
* Saving the trained model using Pickle or Joblib
* Building a more advanced dashboard
* Deploying the Streamlit app online
* Adding more domain-specific financial features

---

## Conclusion

This project successfully built a bankruptcy prediction system using machine learning.

The final tuned Random Forest model was able to identify bankrupt companies with improved recall and F1-score.

The project demonstrates a complete machine learning workflow, including:

* Data exploration
* Data cleaning
* Feature engineering
* Model training
* Model evaluation
* Hyperparameter tuning
* Web app testing

This project can be used as a strong beginner-to-intermediate data science portfolio project.

---

## Author

Hirun Rajapaksha
