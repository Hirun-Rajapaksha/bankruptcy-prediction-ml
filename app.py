import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# Page configuration
st.set_page_config(
    page_title="Bankruptcy Prediction App",
    page_icon="📊",
    layout="wide"
)


# App title
st.title("Bankruptcy Prediction Web App")
st.write(
    "This web app predicts whether a company is likely to be healthy or bankrupt "
    "using financial ratio data."
)


# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("data/bankruptcy_cleaned.csv")
    return df


df = load_data()


# Convert target if needed
df["Bankrupt"] = df["Bankrupt"].replace({
    "No": 0,
    "Yes": 1,
    "no": 0,
    "yes": 1
})

df["Bankrupt"] = df["Bankrupt"].astype(int)


# Separate features and target
X = df.drop("Bankrupt", axis=1)
y = df["Bankrupt"]


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Train model
@st.cache_resource
def train_model():
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=4,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    return model


model = train_model()


# Model evaluation
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)


# Sidebar
st.sidebar.header("App Navigation")

page = st.sidebar.radio(
    "Choose a section",
    ["Project Overview", "Model Performance", "Make Prediction"]
)


if page == "Project Overview":
    st.header("Project Overview")

    st.write("This project uses financial ratio data to predict company bankruptcy.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Records", df.shape[0])

    with col2:
        st.metric("Total Features", X.shape[1])

    with col3:
        bankrupt_count = int(df["Bankrupt"].sum())
        st.metric("Bankrupt Companies", bankrupt_count)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())


elif page == "Model Performance":
    st.header("Model Performance")

    st.write("The web app uses a tuned Random Forest model.")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Accuracy", f"{accuracy:.4f}")

    with col2:
        st.metric("Precision", f"{precision:.4f}")

    with col3:
        st.metric("Recall", f"{recall:.4f}")

    with col4:
        st.metric("F1 Score", f"{f1:.4f}")

    st.info(
        "Because the dataset is highly imbalanced, recall and F1-score are more important "
        "than accuracy alone."
    )


elif page == "Make Prediction":
    st.header("Make a Bankruptcy Prediction")

    st.write(
        "Enter values for selected financial features. Other feature values will be filled "
        "using the dataset median values."
    )

    # Use median row as default input
    input_data = X.median().to_frame().T

    # Select important features for user input
    selected_features = [
        "Persistent_EPS_Last_4_Seasons",
        "Net_Income_to_Total_Assets",
        "Borrowing_Dependency",
        "ROA_C_Before_Interest_Depreciation",
        "Debt_Ratio_Percent",
        "Net_Worth_to_Assets",
        "AvgProfitability",
        "AvgLeverage",
        "AvgLiquidity"
    ]

    # Keep only features that exist in the dataset
    selected_features = [feature for feature in selected_features if feature in X.columns]

    st.subheader("Input Financial Values")

    for feature in selected_features:
        input_data.loc[0, feature] = st.number_input(
            label=feature,
            value=float(input_data.loc[0, feature]),
            format="%.6f"
        )

    if st.button("Predict Bankruptcy Risk"):
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        st.subheader("Prediction Result")

        if prediction == 1:
            st.error("Prediction: Bankrupt / High Risk")
        else:
            st.success("Prediction: Healthy / Low Risk")

        st.write(f"Bankruptcy probability: **{probability:.4f}**")

        if probability >= 0.7:
            st.warning("The company shows high bankruptcy risk.")
        elif probability >= 0.4:
            st.info("The company shows moderate bankruptcy risk.")
        else:
            st.success("The company shows low bankruptcy risk.")