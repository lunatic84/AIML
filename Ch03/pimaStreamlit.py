import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_recall_curve, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns

# 세션 상태 초기화
if "prediction" not in st.session_state:
    st.session_state["prediction"] = None

# 애플리케이션 제목
st.title("Pima Indians Diabetes Prediction App")

# 파일 업로드
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("File Uploaded Successfully!")

    # 데이터 요약
    if st.checkbox("Show Raw Data"):
        st.write(data)

    # min 값이 0인 피처 전처리
    for col in data.columns[:-1]:  # 마지막 열은 타겟
        if data[col].min() == 0 and col != "Pregnancies":  # 예외로 두고 싶은 피처 추가
            mean_value = data[col][data[col] != 0].mean()
            data[col] = data[col].replace(0, mean_value)

    st.write("Data Preprocessed (0 values replaced with mean):")
    st.write(data.describe())

    # 데이터 분할
    X = data.iloc[:, :-1]  # 피처
    y = data.iloc[:, -1]   # 타겟
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 피처 중요도 분석
    if st.checkbox("Show Feature Importance"):
        rf_model = RandomForestClassifier(random_state=42)
        rf_model.fit(X_train, y_train)
        feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': rf_model.feature_importances_
        }).sort_values(by='Importance', ascending=False)

        st.write(feature_importance)
        # 중요도 시각화
        plt.figure(figsize=(10, 5))
        sns.barplot(x='Importance', y='Feature', data=feature_importance)
        plt.title("Feature Importance")
        st.pyplot(plt)

    # 피처 선택
    selected_features = st.multiselect(
        "Select Features for Prediction",
        options=X.columns,
        default=X.columns
    )

    if selected_features:
        X_train_selected = X_train[selected_features]
        X_test_selected = X_test[selected_features]

        # 분류기 선택
        classifier_name = st.selectbox(
            "Select Classifier",
            ["Logistic Regression", "Random Forest", "Decision Tree"]
        )

        if classifier_name == "Logistic Regression":
            model = LogisticRegression()
        elif classifier_name == "Random Forest":
            model = RandomForestClassifier()
        elif classifier_name == "Decision Tree":
            model = DecisionTreeClassifier()

        model.fit(X_train_selected, y_train)
        y_pred = model.predict(X_test_selected)
        accuracy = accuracy_score(y_test, y_pred)
        st.write(f"Accuracy of {classifier_name}: {accuracy:.2f}")

        # 정밀도-재현율 그래프
        if st.checkbox("Show Precision-Recall vs Threshold Curve"):
            y_scores = model.predict_proba(X_test_selected)[:, 1]
            precision, recall, thresholds = precision_recall_curve(y_test, y_scores)

            plt.figure(figsize=(10, 6))
            plt.plot(thresholds, precision[:-1], label="Precision", marker='.')
            plt.plot(thresholds, recall[:-1], label="Recall", marker='.')
            plt.xlabel("Threshold")
            plt.ylabel("Precision / Recall")
            plt.title("Precision and Recall vs Threshold")
            plt.legend()
            st.pyplot(plt)

        # AUC-ROC 그래프
        if st.checkbox("Show AUC-ROC Curve"):
            y_scores = model.predict_proba(X_test_selected)[:, 1]
            fpr, tpr, thresholds = roc_curve(y_test, y_scores)
            roc_auc = auc(fpr, tpr)

            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, marker='.', label=f"AUC = {roc_auc:.2f}")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("ROC Curve")
            plt.legend()
            st.pyplot(plt)

        # 사용자 입력값으로 예측
        st.write("Make a New Prediction")
        new_data = []
        for feature in selected_features:
            value = st.slider(f"Enter value for {feature}", float(X[feature].min()), float(X[feature].max()))
            new_data.append(value)

        if st.button("Predict"):
            # 예측 값을 세션 상태에 저장
            st.session_state["prediction"] = model.predict([new_data])[0]

        # 예측 결과 표시 (세션 상태를 활용)
        if st.session_state["prediction"] is not None:
            st.write(f"Prediction: {'Diabetic' if st.session_state['prediction'] == 1 else 'Non-Diabetic'}")


