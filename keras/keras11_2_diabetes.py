from sklearn.datasets import fetch_california_housing, load_diabetes
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy as np

#1.데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.7, random_state=42)
print(x.shape, y.shape) #(442,10) (442,)

#[실습] 맹그러봐요!! 

#2. 모델 구성 
model = Sequential()
model.add(Dense(7,input_dim=10))
model.add(Dense(8))
model.add(Dense(1))

#3.컴파일 훈련
model.compile(loss='mse',optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=2)

#4.평가예측
loss = model.evaluate(x_test,y_test)
print('loss :', loss)
# results = model.predict(x_test)
# print(results)


