from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.datasets import boston_housing

#1. 데이터 
(x_train,y_train),(x_test,y_test)=boston_housing.load_data()
print(x_train.shape,x_test.shape) #(404, 13) (102, 13)
print(y_train.shape,y_test.shape) #(404,) (102,)

#2. 모델 구성 
model = Sequential()
model.add(Dense(7, input_dim=13))
model.add(Dense(8))
model.add(Dense(1))

#3.컴파일 훈련
model.compile(loss= 'mse' , optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=2)

print("======================================")
#4. 평가예측 
loss = model.evaluate(x_test,y_test) #bat
print('loss :' , loss)
# results = model.predict(x_test)
# print(results)


# # 여기까지 9/2 수요일 day3
# 어제꺼 복습 

# 1. 로스 = 에러 = 오차 = cost(비용/코스트)
# 2. mse :평균 제곱 오차 (mean squared error)