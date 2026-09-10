# 19-1 카피

from sklearn.datasets import fetch_california_housing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy as np
import time

#1. 데이터 
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) #(20640,8) (20640,)

"""
MinMacScler : 최솟값은 0, 최댓값은 1로 만들고 나머지는 그 사이에 배치하는 것!

 원값 - Min
-------------
 Max - Min

"""
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x) #fit은 실행하다 
x = scaler.transform(x)

print(x)
print(np.min(x),np.max(x))  #최소값과 최대값
# 0.0   / 1.0000000000000002 <- 파이썬 문제 걍 0으로 인식 

a = 0.1
b = 0.2
print(a+b) #0.30000000000000004

# exit()
x_train, x_test, y_train, y_test = train_test_split(
    x,y ,train_size= 0.75
     , random_state=4333
     )

x_train, x_val, y_train, y_val =train_test_split(
    x_train,y_train,
    train_size = 0.5
)

#2. 모델 구성 
model = Sequential()
model.add(Dense(7,input_dim=8))
model.add(Dense(8,activation='relu'))
model.add(Dense(1))

#3. 컴파일 훈련
model.compile(loss='mse',optimizer = 'adam')
hist = model.fit(x_train,y_train,epochs=50,batch_size=10,
          validation_split = 0.2)

#평가예측 
loss = model.evaluate(x_test,y_test)
print('loss : ', loss)
results = model.predict(x_test)
print(results)














