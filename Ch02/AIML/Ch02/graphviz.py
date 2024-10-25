from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt

# Load the iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Train a decision tree classifier with max_depth=2, min_samples_split=3
clf = DecisionTreeClassifier(max_depth=10, min_samples_split=4, min_samples_leaf=10)
clf.fit(X, y)

# Plot the decision tree
plt.figure(figsize=(12,8))
tree.plot_tree(clf, filled=True, feature_names=iris.feature_names, class_names=iris.target_names, rounded=True)
plt.show()
