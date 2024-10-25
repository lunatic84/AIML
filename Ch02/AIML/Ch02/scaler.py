from sklearn.datasets import load_iris
import pandas as pd
# 붓꽃 데이터 세트를 로딩하고 DataFrame으로 변환
iris = load_iris()
iris_data = iris.data
iris_df = pd.DataFrame(data = iris_data, columns = iris.feature_names)

print("feature 들의 평균 값")
print(iris_df.mean())
print("\nfeature 들의 분산 값")
print(iris_df.var())

from sklearn.preprocessing import StandardScaler

# StandardScaler객체 생성
scaler = StandardScaler()
# StandardScaler로 데이터 세트 변환. fit()과 transform()호출
# fit() : 데이터의 통계적 정보 학습
# transform() : fit()에서 학습한 통계적 정보를 이용하여 데이터 변환
scaler.fit(iris_df)
iris_scaled = scaler.transform(iris_df) 
# print(type(iris_scaled))

# transform() 시 스케일 변환된 데이터 세트가 Numpy ndarray로 
# 반환돼 이를 DataFrame으로 변환
iris_df_scaled = pd.DataFrame(data = iris_scaled, columns = iris.feature_names)
print("feature들의 평균 값")
print(iris_df_scaled.mean())
print("\nfeature들의 분산 값")
print(iris_df_scaled.var())

from sklearn.preprocessing import MinMaxScaler

# MinMaxScaler 객체 생성
scaler = MinMaxScaler()
# MinMaxScaler로 데이터 세트 변환
# fit()과 transform() 호출
scaler.fit(iris_df)
iris_scaled = scaler.transform(iris_df)

# DataFrame으로 변환
iris_df_scaled = pd.DataFrame(data=iris_scaled, columns=iris.feature_names)
print("feature들의 최소값")
print(iris_df_scaled.min())
print("feature들의 최대값")
print(iris_df_scaled.max())

from sklearn.preprocessing import MinMaxScaler
import numpy as np

# 학습 데이터는 0부터 10까지, 
# 테스트 데이터는 0부터 5까지 값을 가지는 데이터 세트로 생성
# Scaler 클래스의 fit(), transform()은 2차원 이상 데이터만 가능하므로
# reshape(-1, 1)로 차원 변경
train_array = np.arange(0, 11).reshape(-1, 1)
test_array = np.arange(0, 6).reshape(-1, 1)
# MinMaxScaler 객체에 별도의 feature_range 파라미터 값을
# 지정하지 않으면 0~1 값으로 변환
scaler = MinMaxScaler()
# fit()하게 되면 train_array 데이터의 최소값이 0, 최대값이 10으로 설정
scaler.fit(train_array)
# 1/10 scale로 train_array 데이터 변환
train_scaled = scaler.transform(train_array)
print("원본 train_array 데이터:", np.round(train_array.reshape(-1), 2))
print("Scale된 train_array 데이터:", np.round(train_scaled.reshape(-1), 2))

# MinMaxScaler에 test_array를 fit()하게 되면 원본 데이터의 
# 최소값이 0, 최대값이 5로 설정됨
scaler.fit(test_array)
test_scaled = scaler.transform(test_array)
print("원본 train_array 데이터:", np.round(test_array.reshape(-1), 2))
print("Scale된 train_array 데이터:", np.round(test_scaled.reshape(-1), 2))
# 1이 0.2로 train의 경우 1이 0.1 따라서 데이터의 스케일링이 맞지 않음

scaler = MinMaxScaler()
scaler.fit(train_array)
train_scaled = scaler.transform(train_array)
print("원본 train_array 데이터:", np.round(train_array.reshape(-1), 2))
print("Scale된 train_array 데이터:", np.round(train_scaled.reshape(-1), 2))

test_scaled = scaler.transform(test_array)
print("원본 test_array 데이터:", np.round(test_array.reshape(-1), 2))
print("Scale된 test_array 데이터:", np.round(test_scaled.reshape(-1), 2))