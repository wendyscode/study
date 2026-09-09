# 17-1 카피

#실습 : 발리데이션 맹그러보기 

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
print(x.shape, y.shape)

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

# print("걸린시간 :", round(end_time - start_time,2), "초")

print("===================history========================")
print(hist)
print("===================hist.history========================")
print(hist.history)
print("===================loss========================")
print(hist.history["loss"])
print("===================val_loss========================")
print(hist.history["val_loss"])
print("===========================================")


####그래프그리기####
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Malgun Gothic'       #한글깨짐현상 폰트지정해주기 

plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'][2:],c='red',label='loss')# y값만 넣으면 시간순으로 그려줌
plt.plot(hist.history['val_loss'][2:],c='blue',label='val_loss')
plt.legend(loc= 'upper right')#우측상단에 라벨표시  #location 위치어디로할건지 
plt.title('캘리포니아 Loss') 
plt.xlabel('epoch')
plt.ylabel('loss')
plt.grid() #격자표시 추가 
plt.show()

# 3 애포 부터 100애포 까지 












