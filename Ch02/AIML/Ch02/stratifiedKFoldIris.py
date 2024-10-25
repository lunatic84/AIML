from sklearn.model_selection import StratifiedKFold
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

iris = load_iris()
features = iris.data
label = iris.target

dt_clf = DecisionTreeClassifier(random_state = 156)

skfold = StratifiedKFold(n_splits=3)
n_iter = 0
cv_accuracy = []

# StratifiedKFold의 split( ) 호출시 
# 반드시 레이블 데이터 세트도 추가 입력 필요
for train_idx, test_idx in skfold.split(features, label):
    # split( )으로 반환된 인덱스를 
    # 이용해 학습용, 검증용 테스트 데이터 추출
    X_train, X_test = features[train_idx], features[test_idx]
    y_train, y_test = label[train_idx], label[test_idx]
    # 학습 및 예측
    dt_clf.fit(X_train, y_train)
    pred = dt_clf.predict(X_test)
    # 반복 시마다 정확도 측정
    n_iter += 1
    accuracy = np.round(accuracy_score(y_test, pred), 4)
    train_size = X_train.shape[0]
    test_size = X_test.shape[0]
    print("\n#{0} 교차 검증 정확도: {1}, 학습데이터 크기: {2}, 검증 데이터 크기: {3}"
          .format(n_iter, accuracy, train_size, test_size))
    cv_accuracy.append(accuracy)

# 교차 검증별 정확도 및 평균 정확도 계산
print("\n## 교차 검증별 정확도:", np.round(cv_accuracy, 4))
print("## 평균 검증 정확도:", np.round(np.mean(cv_accuracy), 4))
