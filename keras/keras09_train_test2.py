# train/test 슬라이싱으로 나누기
# 09 카피

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10]) 

# x_train = np.array([1,2,3,4,5,6,7])
# y_train = np.array([1,2,3,4,5,6,7])

# x_test = np.array([8,9,10])
# y_test = np.array([8,9,10])

#[찾아보기] 넘파이 리스트의 슬라이싱 => 7:3 으로 나누자

x_train= x[:7]   # x[:7] = x[0:7]
y_train= y[:7]   # y[:7] = y[0:7] 

x_test = x[7:]   # x[7:] = x[7:10] 
y_test = y[7:]   # y[7:] = y[7:10]


#2. 모델 구성
model = Sequential()
model.add(Dense(1, input_dim=1))
model.add(Dense(5))
model.add(Dense(7))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=500, batch_size=2)   # 1epoch에 bathc 4로 한번 batch 3으로 한번 총 두번돌음

#4. 추론, 예측
loss = model.evaluate(x_test, y_test)
print("loss: ", loss)
# results = model.predict(x_test, y_test)
# print("results: ", results)