from sklearn.preprocessing import OneHotEncoder
import numpy as np
items = ['TV','냉장고','전자레인지','컴퓨터',
         '선풍기','선풍기','믹서','믹서']
# 2차원 ndarray로 변환
items = np.array(items).reshape(-1, 1)
# print(items)
# 원-핫 인코딩 적용
oh_encoder = OneHotEncoder()
oh_encoder.fit(items)
oh_labels = oh_encoder.transform(items)
# print(oh_labels), CSR(Compressed Sparse Row) 방식
# OneHotEncoder로 변환한 결과는 희소행렬이므로 
# toarray()를 이용해 밀집 행렬로 변환
print('원-핫 인코딩 데이터')
print(oh_labels.toarray())
print('원-핫 인코딩 데이터 차원')
print(oh_labels.shape)

