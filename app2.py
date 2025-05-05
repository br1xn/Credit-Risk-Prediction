# Importing Modules
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load models and other resources
logistic_model = joblib.load("logistic_model.pkl")
rf_model = joblib.load("random_forest.pkl")
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("features.pkl")

# App Title
st.title("\U0001F4B3 Credit Risk Prediction App")
st.write("Predict whether a customer is a Good or Bad credit risk using two models.")

# Sidebar for inputs
st.sidebar.header("\U0001F4DD Applicant Information")
age = st.sidebar.slider("Age", min_value=18, max_value=75, value=30)
job = st.sidebar.selectbox("Job Type", [0, 1, 2, 3])
credit_amount = st.sidebar.number_input("Credit Amount", min_value=100)
duration = st.sidebar.slider("Duration (in months)", 4, 72, 12)
sex = st.sidebar.selectbox("Sex", ['male', 'female'])
housing = st.sidebar.selectbox("Housing Type", ['own', 'rent', 'free'])
saving = st.sidebar.selectbox("Saving Account", ["moderate", "quite rich", "rich", "unkown"])
checking = st.sidebar.selectbox("Checking Account", ['moderate', 'rich', 'unkown'])
purpose = st.sidebar.selectbox("Purpose", [
    'car', 'domestic appliances', 'education', 'furniture/equipment',
    'radio/TV', 'repairs', 'vacation/others']
)

# Input Processing
input_dict = {
    'Age': age,
    'Job': job,
    'Credit amount': credit_amount,
    'Duration': duration,
    'Sex_male': sex == 'male',
    'Housing_own': housing == 'own',
    'Housing_rent': housing == 'rent',
    'Saving accounts_moderate': saving == 'moderate',
    'Saving accounts_quite rich': saving == 'quite rich',
    'Saving accounts_rich': saving == 'rich',
    'Saving accounts_unkown': saving == 'unkown',
    'Checking account_moderate': checking == 'moderate',
    'Checking account_rich': checking == 'rich',
    'Checking account_unkown': checking == 'unkown',
    'Purpose_car': purpose == 'car',
    'Purpose_domestic appliances': purpose == 'domestic appliances',
    'Purpose_education': purpose == 'education',
    'Purpose_furniture/equipment': purpose == 'furniture/equipment',
    'Purpose_radio/TV': purpose == 'radio/TV',
    'Purpose_repairs': purpose == 'repairs',
    'Purpose_vacation/others': purpose == 'vacation/others'
}

input_df = pd.DataFrame([input_dict])
input_df = input_df.reindex(columns=feature_columns, fill_value=0)
input_scaled = scaler.transform(input_df)

# Predict
if st.button("Predict Credit Risk"):
    log_pred = logistic_model.predict(input_scaled)[0]
    log_prob = logistic_model.predict_proba(input_scaled)[0][log_pred]

    rf_pred = rf_model.predict(input_scaled)[0]
    rf_prob = rf_model.predict_proba(input_scaled)[0][rf_pred]

    st.markdown("## \U0001F4CA Prediction Results")
    # Logistic Regression result
    if log_pred == 1:
        st.success(f"✅ **Logistic Regression Prediction:** Good Credit Risk\nConfidence: **{log_prob:.2%}**")
    else:
        st.error(f"⚠️ **Logistic Regression Prediction:** Bad Credit Risk\nConfidence: **{log_prob:.2%}**")

    # Random Forest result
    if rf_pred == 1:
        st.success(f"✅ **Random Forest Prediction:** Good Credit Risk\nConfidence: **{rf_prob:.2%}**")
    else:
        st.error(f"⚠️ **Random Forest Prediction:** Bad Credit Risk\nConfidence: **{rf_prob:.2%}**")

    # Insights Section
    st.markdown("---")
    st.markdown("## \U0001F4D6 Model Interpretation")

    # Logistic Coefficients
    st.markdown("### Logistic Regression - Feature Influence")
    coefs = logistic_model.coef_[0]
    coef_df = pd.DataFrame({"Feature": feature_columns, "Coefficient": coefs})
    coef_df = coef_df.sort_values("Coefficient")

    fig1, ax1 = plt.subplots(figsize=(8, 6))
    sns.barplot(x="Coefficient", y="Feature", data=coef_df, ax=ax1, palette="coolwarm")
    ax1.axvline(0, color='black', linewidth=0.8)
    st.pyplot(fig1)

    # Random Forest Importances
    st.markdown("### Random Forest - Feature Importance")
    importances = rf_model.feature_importances_
    imp_df = pd.DataFrame({"Feature": feature_columns, "Importance": importances})
    imp_df = imp_df.sort_values("Importance", ascending=False).head(10)

    fig2, ax2 = plt.subplots(figsize=(8, 6))
    sns.barplot(x="Importance", y="Feature", data=imp_df, ax=ax2, palette="viridis")
    st.pyplot(fig2)

    # Strategy Box
    st.markdown("---")
    st.markdown("## \U0001F4C8 Actionable Insights")
    st.markdown("- Ensure **stable housing** and **moderate savings** to reduce credit risk.")
    st.markdown("- Be cautious with **high credit amounts** and **long durations**.")
    st.markdown("- Applicants with **no checking account info** are more likely to be high risk.")
