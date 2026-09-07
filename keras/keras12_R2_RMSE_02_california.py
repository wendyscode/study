# R2 기준 : 0.55 이상 올려보기 

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import numpy as np

#1. 데이터 
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) 

x_train, x_test, y_train, y_test = train_test_split(x,y ,train_size=.07, random_state=42)

# 2.모델구성 
model = Sequential()
model.add(Dense(6,input_dim=8))
model.add(Dense(7))
model.add(Dense(8))
model.add(Dense(1))

# 3. 컴파일 훈련
model.compile(loss='mse' , optimizer='adam')
model.fit(x_train, y_train, epochs=100 , batch_size=3)

print("======================================")

# 4. 평가, 예측 
loss = model.evaluate(x_test,y_test) 
print('loss :', loss)

y_predict = model.predict(x_test)
from sklearn.metrics import r2_score,mean_squared_error
r2 = r2_score(y_test, y_predict)
print("r2 : ", r2)