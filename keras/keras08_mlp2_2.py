import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array(range(10)) #range(10) = 0~9
print(x)

x = np.array(range(1,10))
print(x) # [1 2 3 4 5 6 7 8 9]

x = np.array(range(1,11))
print(x) # [1 2 3 4 5 6 7 8 9 10]   

x = np.array([range(10), range(21,31,), range(201,211)]).T   #transpose로 행열 변경
print(x.shape) #(3, 10) -> (10, 3)으로 변경됨

y = np.array(range(1,11))
print(y.shape) #(10,)

#2. 모델구성
#[실습]
#[10, 31, 211] 을 찾아봐아라  / 11.00까지 뜨면 합격 
model = Sequential()
model.add(Dense(5, input_dim=3))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse',optimizer="adam")
model.fit(x,y, epochs=10000, batch_size=2)

#4. 평가, 예측
results = model.predict(np.array([[10,31,211]]))
print('results : ', results)  


