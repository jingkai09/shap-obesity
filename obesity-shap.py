import streamlit as st
import shap
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
from streamlit_shap import st_shap

# Load the dataset
obesity_data = pd.read_csv("ObesityDataSet.csv")

# Preprocessing
label_encoders = {}
for column in obesity_data.select_dtypes(include='object').columns:
    le = LabelEncoder()
    obesity_data[column] = le.fit_transform(obesity_data[column])
    label_encoders[column] = le

# Separate features and target variable
X = obesity_data.drop('NObeyesdad', axis=1)
y = obesity_data['NObeyesdad']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model training
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train_scaled, y_train)

# Make predictions
y_pred = clf.predict(X_test_scaled)

# SHAP explainer
explainer = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(X_test_scaled)

# Streamlit app
st.title("SHAP Analysis for Obesity Prediction")

# Part 1: General SHAP Analysis
st.header("Part 1: General SHAP Analysis")
st.dataframe(pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose())

# Summary plot
st.subheader("Summary Plot")
fig, ax = plt.subplots()
shap.summary_plot(shap_values, X_test_scaled, feature_names=X.columns, show=False)
st.pyplot(fig)

# Summary plot for each class
st.subheader("Summary Plot for Each Class")
for i in range(len(clf.classes_)):
    st.subheader(f"Summary Plot for Class {label_encoders['NObeyesdad'].inverse_transform([i])[0]}")
    fig, ax = plt.subplots()
    shap.summary_plot(shap_values[i], X_test_scaled, feature_names=X.columns, show=False)
    st.pyplot(fig)

# Part 2: Individual Input Prediction & Explanation
st.header("Part 2: Individual Input Prediction & Explanation")

# Input fields for features
input_data = {}
for feature in X.columns:
    input_data[feature] = st.number_input(f"Enter {feature}:", value=float(X_test[feature].mean()))

# Create a DataFrame from input data
input_df = pd.DataFrame(input_data, index=[0])
input_df_scaled = scaler.transform(input_df)

# Make prediction
prediction = clf.predict(input_df_scaled)[0]
probability = clf.predict_proba(input_df_scaled)[0]

# Display prediction
st.write(f"**Prediction:** {label_encoders['NObeyesdad'].inverse_transform([prediction])[0]}")
st.write(f"**Prediction Probabilities:**")
st.write(dict(zip(label_encoders['NObeyesdad'].classes_, probability)))

# SHAP explanation for the input
shap_values_input = explainer.shap_values(input_df_scaled)

# Force plot
st.subheader("Force Plot")
st_shap(shap.force_plot(explainer.expected_value[prediction], shap_values_input[prediction], input_df_scaled), height=400, width=1000)

# Decision plot
st.subheader("Decision Plot")
st_shap(shap.decision_plot(explainer.expected_value[prediction], shap_values_input[prediction], input_df_scaled))
