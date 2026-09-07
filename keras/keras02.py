from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터
x=np.array([1,2,3,4,5,6])
y=np.array([1,2,3,4,5,6])

#2. 모델구성
model = Sequential()
model.add(Dense(1, input_dim=1))

#3.컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x,y, epochs=200)

#4. 평가 예측
loss = model.evaluate(x, y) #로스 = 에러 = 오차 같은말! 
print("lose :" , loss)
result = model.predict(np.array([1,2,3,4,5,6,7]))  #2개이상은 리스트!
print("7의 예측값 : ", result)