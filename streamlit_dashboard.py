import pickle

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.metrics import classification_report, roc_auc_score, roc_curve


def commission_binning(X):
    return (X > 0).astype(int)


with open("model.pkl", "rb") as file:
    saved_objects = pickle.load(file)
    model = saved_objects["model"]
    preprocessor = saved_objects["preprocessor"]
    y_test_data = saved_objects["y_test"]
    y_pred_data = saved_objects["y_pred"]

st.title("Insurance Claim Prediction Dashboard")
st.markdown(
    "This app predicts the likelihood of an insurance claim using a logistic regression model."
)

st.header("Input Features")

agency_type = st.selectbox("Agency Type", ["Travel Agency", "Airlines"])

agency = st.selectbox(
    "Agency",
    [
        "C2B",
        "EPX",
        "JZI",
        "CWT",
        "LWC",
        "ART",
        "CSR",
        "SSI",
        "RAB",
        "KML",
        "TST",
        "TTW",
        "JWT",
        "ADM",
        "CCR",
        "CBH",
    ],
)

distribution_channel = st.selectbox("Distribution Channel", ["Online", "Offline"])

product_name = st.selectbox(
    "Product Name",
    [
        "Annual Silver Plan",
        "Cancellation Plan",
        "Basic Plan",
        "2 way Comprehensive Plan",
        "Bronze Plan",
        "1 way Comprehensive Plan",
        "Rental Vehicle Excess Insurance",
        "Single Trip Travel Protect Gold",
        "Silver Plan",
        "Value Plan",
        "24 Protect",
        "Annual Travel Protect Gold",
        "Comprehensive Plan",
        "Ticket Protector",
        "Travel Cruise Protect",
        "Single Trip Travel Protect Silver",
        "Individual Comprehensive Plan",
        "Gold Plan",
        "Annual Gold Plan",
        "Child Comprehensive Plan",
        "Premier Plan",
        "Annual Travel Protect Silver",
        "Single Trip Travel Protect Platinum",
        "Annual Travel Protect Platinum",
        "Spouse or Parents Comprehensive Plan",
        "Travel Cruise Protect Family",
    ],
)

duration = st.number_input("Duration (in days)", min_value=1, step=1)

destination = st.text_input("Destination (City)", value="")

net_sales = st.number_input(
    "Net Sales (in USD)", min_value=0.0, step=0.01, format="%.2f"
)

age = st.number_input("Age", min_value=0, step=1)

commission = st.number_input(
    "Commission (in value)", min_value=0.0, step=0.01, format="%.2f"
)

if st.button("Predict"):
    input_data = {
        "Agency Type": agency_type,
        "Agency": agency,
        "Distribution Channel": distribution_channel,
        "Product Name": product_name,
        "Duration": duration,
        "Destination": destination,
        "Net Sales": net_sales,
        "Age": age,
        "Commision (in value)": commission,
    }

    input_df = pd.DataFrame([input_data])

    input_transformed = preprocessor.transform(input_df)

    y_pred = model.predict(input_transformed)
    y_pred_proba = model.predict_proba(input_transformed)[:, 1]

    st.subheader("Prediction")
    prediction = (
        "Likely to file a claim" if y_pred[0] == 1 else "Not likely to file a claim"
    )
    st.write(f"The model predicts: **{prediction}**")
    st.write(f"Probability of claim: **{y_pred_proba[0]:.2f}**")

    st.subheader("Classification Report")
    report = classification_report(y_test_data, y_pred_data, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df)

    st.subheader("ROC AUC Score")
    roc_auc = roc_auc_score(y_test_data, y_pred_data)
    st.write(f"ROC AUC Score: **{roc_auc:.2f}**")

    st.subheader("ROC Curve")
    fpr, tpr, _ = roc_curve(y_test_data, y_pred_data)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    st.pyplot(plt)
