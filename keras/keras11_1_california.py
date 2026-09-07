# import
# ssl_create_default_https_context = ssl_create_default_https_context

from sklearn.datasets import fetch_california_housing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy as np

#1. 데이터 
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target #분리가 된다 
print(x.shape, y.shape) #(20640, 8) (20640,) / 20640개의데이터 8개의종류 (방의갯수,,등등)
#행무시열우선 input_dim이 8이다!! .shape로 input_dim찾기

x_train, x_test, y_train, y_test = train_test_split(x,y ,train_size=.07, random_state=42)

#2. 모델 구성 
model = Sequential()
model.add(Dense(7,input_dim=8))
model.add(Dense(8))
model.add(Dense(1))

#3. 컴파일 훈련
model.compile(loss='mse',optimizer = 'adam')
model.fit(x_train,y_train,epochs=100,batch_size=2)

#평가예측 
loss = model.evaluate(x_test,y_test)
print('loss : ', loss)
results = model.predict(x_test)
print(results)

# #그래프 그리기
# import matplotlib.pyplot as plt
# plt.scatter(x,y)
# plt.plot(x_test,result,color='yellow')
# plt.show()












