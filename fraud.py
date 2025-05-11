import streamlit as st
import pandas as pd
import numpy as np
from collections import Counter

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE

import matplotlib.pyplot as plt
import seaborn as sns

# Load and prepare data
@st.cache_data
def load_data():
    df = pd.read_csv('creditcard.csv')
    df = df.dropna().drop_duplicates()
    scaler = StandardScaler()
    df['Amount_scaled'] = scaler.fit_transform(df[['Amount']])
    df.drop(['Amount', 'Time'], axis=1, inplace=True)
    X = df.drop('Class', axis=1)
    y = df['Class']
    return train_test_split(X, y, test_size=0.2, random_state=42)

X_train, X_test, y_train, y_test = load_data()
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Model dictionary
model_dict = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'KNN': KNeighborsClassifier(),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'SVC': SVC(probability=True)
}

def evaluate_model(name, model):
    model.fit(X_train_resampled, y_train_resampled)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    return {
        "Confusion Matrix": confusion_matrix(y_test, y_pred),
        "Classification Report": classification_report(y_test, y_pred, output_dict=True),
        "ROC AUC Score": roc_auc_score(y_test, y_proba),
        "FPR": fpr,
        "TPR": tpr
    }

# Streamlit UI
st.title("Fraud Detection: Train & Compare Models")
option = st.radio("Select a mode:", ("Train Single Model", "Compare Models"))

if option == "Train Single Model":
    selected_model_name = st.selectbox("Select model to train", list(model_dict.keys()))
    if st.button("Train"):
        with st.spinner("Training..."):
            results = evaluate_model(selected_model_name, model_dict[selected_model_name])

        st.subheader(f"{selected_model_name} Results")

        # Confusion Matrix - visual
        st.markdown("**Confusion Matrix:**")
        fig_cm, ax_cm = plt.subplots()
        sns.heatmap(results["Confusion Matrix"], annot=True, fmt='d', cmap='Blues', ax=ax_cm)
        ax_cm.set_xlabel("Predicted")
        ax_cm.set_ylabel("Actual")
        st.pyplot(fig_cm)

        # Classification Report - table
        st.markdown("**Classification Report:**")
        report_df = pd.DataFrame(results["Classification Report"]).transpose()
        st.dataframe(report_df.style.format(precision=2))

        # ROC AUC Score
        st.markdown(f"**ROC AUC Score:** `{results['ROC AUC Score']:.4f}`")

        # ROC Curve
        st.markdown("**ROC Curve:**")
        fig_roc, ax_roc = plt.subplots()
        ax_roc.plot(results["FPR"], results["TPR"], label=f"AUC = {results['ROC AUC Score']:.4f}")
        ax_roc.plot([0, 1], [0, 1], 'k--', label="Random")
        ax_roc.set_xlabel("False Positive Rate")
        ax_roc.set_ylabel("True Positive Rate")
        ax_roc.set_title("ROC Curve")
        ax_roc.legend()
        st.pyplot(fig_roc)

elif option == "Compare Models":
    selected_models = st.multiselect("Select models to compare", list(model_dict.keys()))
    if st.button("Compare") and selected_models:
        st.subheader("Model Comparison")
        for name in selected_models:
            with st.spinner(f"Training {name}..."):
                results = evaluate_model(name, model_dict[name])
            st.markdown(f"### {name}")
            
            # Confusion Matrix - as text
            st.markdown("**Confusion Matrix:**")
            st.text(results["Confusion Matrix"])

            # Classification Report - formatted table
            st.markdown("**Classification Report:**")
            report_df = pd.DataFrame(results["Classification Report"]).transpose()
            st.dataframe(report_df.style.format(precision=2))

            # ROC AUC Score
            st.markdown(f"**ROC AUC Score:** `{results['ROC AUC Score']:.4f}`")
