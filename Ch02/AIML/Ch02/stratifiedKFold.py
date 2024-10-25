from sklearn.model_selection import StratifiedKFold
from sklearn.datasets import load_iris
import pandas as pd

iris = load_iris()
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['label'] = iris.target
skf = StratifiedKFold(n_splits = 3)
n_iter = 0

for train_idx, test_idx in skf.split(iris_df, iris_df['label']):
    n_iter += 1
    label_train = iris_df['label'].iloc[train_idx]
    label_test = iris_df['label'].iloc[test_idx]
    print('## 교차 검증: {0}'.format(n_iter))
    print('학습 레이블 데이터 분포: \n', label_train.value_counts())
    print('검증 레이블 데이터 분호: \n', label_test.value_counts())
    
grid_parameters = {'max_depth': [1,2,3],
                   'min_samples_split':[2,3]
                  }

