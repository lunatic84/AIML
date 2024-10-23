from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_iris
import pandas as pd

# MinMaxScaler객체 생성
scaler = MinMaxScaler()
# MinMaxScaler로 데이터 세트 변환. 
# fit()과 transform() 호출.
scaler.fit()
iris_scaled = scaler.transform()

# transform() 시 스케일 변환된 데이터 세트가
# Numpy ndarray로 반환돼 이를 DataFrame으로 변환
iris_df_scaled = pd.DataFrame(data=iris_scaled, columns=iris.feature_names)
