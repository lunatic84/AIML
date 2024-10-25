from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np

iris_data = load_iris()
X = iris_data.data
y = iris_data.target

X_train, X_test, y_train, y_test = train_test_split(iris_data.data, iris_data.target,\
                                                    test_size = 0.3, stratify = None,
                                                    random_state = 121)
df_clf = DecisionTreeClassifier()
df_clf.fit(X_train, y_train)
pred = df_clf.predict(X_test)
print("예측 정확도 : {0:.4f}".format(accuracy_score(y_test, pred)))


