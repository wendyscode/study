import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터 
# x = np.array([[1,2,3,4,5], # 잘못된 데이터 
#             [6,7,8,9,10]])
x = np.array([[1,6],[2,7],[3,8],[4,9],[5,10]]) #올바른데이터
y = np.array([1,2,3,4,5])

print("x.shape : ", x.shape) # (5, 2) : 5개짜리 2개
print("y.shape : ", y.shape) # (5,) : 5개짜리 1개

#2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=2))
model.add(Dense(7))
model.add(Dense(3))
model.add(Dense(1))

#3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x,y,epochs=100, batch_size=3) #batch_size=1 : 1개씩 잘라서 훈련한다

#4. 평가, 예측
loss = model.evaluate(x,y)
print('loss : ', loss)
results = model.predict(np.array([[6,11]]))      #(1,2)
print("[6,11]의 예측값 : ", results)

#행무시 열우선 