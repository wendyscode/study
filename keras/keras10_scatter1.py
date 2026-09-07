import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,7,5,7,8,6,10])

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
)
print('x_train = ', x_train)
print('x_test = ', x_test)
print('y_train = ', y_train)
print('y_test = ', y_test)

#2.모델구성
model = Sequential()
model.add(Dense(7,input_dim=1))
model.add(Dense(8))
model.add(Dense(5))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse',optimizer='adam')
model.fit(x,y,epochs=50, batch_size=2)

print("=============================")
#4. 평가,예측
loss = model.evaluate(x_test,y_test, ) #batch_siz=32)
print('loss : ', loss)
results = model.predict(x)
print(results)

# 그래프 그리기 
import matplotlib.pyplot as plt
plt.scatter(x,y)  #데이터의위치 점으로 표시 
# plt.plot(x,y)#선으로 긋기 
plt.plot(x,results ,color='red')
plt.show() 
