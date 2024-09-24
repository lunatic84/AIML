from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import numpy as np

iris = load_iris()
features = iris.data
label = iris.target
dt_clf = DecisionTreeClassifier(random_state = 156)

kfold = KFold(n_splits = 5) # kfold 객체 생성
cv_accuracy = []
print('붓꽃 데이터 세트 크기:', features.shape[0])

n_iter = 0

# KFold 객체는 split()를 호출하면 학습용/검증용 데이터로 
# 분할할 수 있는 인덱스를 반환
for train_idx, test_idx in kfold.split(features):
    


