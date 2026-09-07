from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터 
x = np.array([1,2,3,4,5])
y = np.array([1,2,4,3,5]) 

#2. 모델구성
model = Sequential()
model.add(Dense(3, input_dim=1)) #하나가들어가서 3이 나왓다
model.add(Dense(5, input_dim=3)) #3이 들어가서 5가 나온다
model.add(Dense(4, input_dim=5)) #5이 들어가서 4가 나온다
model.add(Dense(1, input_dim=4)) #4이 들어가서 1가 나온다



#3.컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x,y, epochs=5000)

#4. 평가, 예측
loss = model.evaluate(x, y)  
print("lose  :" , loss)
result = model.predict(np.array([1,2,3,4,5]))
print("7의 예측값 : ", result)


