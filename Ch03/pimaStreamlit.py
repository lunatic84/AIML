import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_recall_curve, roc_curve, auc

# 애플리케이션 제목
st.title("Pima Indians Diabetes Prediction App")

# 파일 업로드
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("File Uploaded Successfully!")

    # 데이터 확인
    if st.checkbox("Show Raw Data"):
        st.write(data)

    # 데이터 전처리: min 값이 0인 피처 평균값으로 대체
    for col in data.columns[:-1]:
        if data[col].min() == 0 and col != "Pregnancies":  # 'Pregnancies'는 예외 처리
            mean_value = data[col][data[col] != 0].mean()
            data[col] = data[col].replace(0, mean_value)

    # 데이터 확인
    st.write("Processed Data (0 values replaced with mean):")
    st.write(data.describe())

    # 데이터 분할
    X = data.iloc[:, :-1]  # 피처
    y = data.iloc[:, -1]   # 타겟

    # X 데이터가 비어있거나 컬럼이 없을 경우 처리
    if X.empty or X.columns.empty:
        st.error("Error: The dataset is invalid or has no features. Please check your uploaded file.")
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 피처 선택
        st.write("Columns in X:", X.columns)
        selected_features = st.multiselect(
            "Select Features for Prediction",
            options=list(X.columns),
            default=list(X.columns)
        )

        if selected_features:
            X_train_selected = X_train[selected_features]
            X_test_selected = X_test[selected_features]

            # 분류기 선택
            classifier_name = st.selectbox(
                "Select Classifier",
                ["Logistic Regression", "Random Forest", "Decision Tree"]
            )

            # 분류기 초기화
            if classifier_name == "Logistic Regression":
                model = LogisticRegression()
            elif classifier_name == "Random Forest":
                model = RandomForestClassifier()
            elif classifier_name == "Decision Tree":
                model = DecisionTreeClassifier()

            # 모델 학습
            model.fit(X_train_selected, y_train)
            y_scores = model.predict_proba(X_test_selected)[:, 1]

            # Accuracy
            y_pred = (y_scores > 0.5).astype(int)
            accuracy = accuracy_score(y_test, y_pred)
            st.write(f"Accuracy of {classifier_name}: {accuracy:.2f}")

            # Threshold 조정 슬라이더
            threshold = st.slider("Adjust Threshold", 0.0, 1.0, 0.5, 0.01)

            # Threshold에 따른 예측
            y_pred_threshold = (y_scores > threshold).astype(int)
            accuracy_threshold = accuracy_score(y_test, y_pred_threshold)
            st.write(f"Accuracy with threshold {threshold:.2f}: {accuracy_threshold:.2f}")

            # 정밀도-재현율 그래프
            if st.checkbox("Show Precision-Recall vs Threshold Curve"):
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
                prediction = (model.predict_proba([new_data])[:, 1] > threshold).astype(int)
                st.write(f"Prediction with threshold {threshold:.2f}: {'Diabetic' if prediction[0] == 1 else 'Non-Diabetic'}")
