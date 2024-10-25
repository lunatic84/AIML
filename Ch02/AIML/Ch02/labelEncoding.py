from sklearn.preprocessing import LabelEncoder

items = ['TV', '냉장고', '전자레인지', '컴퓨터',
          '선풍기', '선풍기', '믹서', '믹서']

# LabelEncoder를 객체로 생성한 후, 
# fit()과 transform()으로 레이블 인코딩 수행
encoder = LabelEncoder()
encoder.fit(items)
labels = encoder.transform(items)
print('인코딩 변환값:', labels)
# 0 1 4 5 3 3 2 2
print('인코딩 클래스:', encoder.classes_)
# TV:0, 냉장고:1, 믹서:2, 선풍기:3, 전자레인지:4, 컴퓨터:5
print('디코딩 원본값:', encoder.inverse_transform([4,5,2,0,1,1,3,3]))
# 전자레인지, 컴퓨터, 믹서, TV, 냉장고, 냉장고, 선풍기, 선풍기

