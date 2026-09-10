# 27 카피

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


# exit()
x_train, x_test, y_train, y_test = train_test_split(
    x,y ,train_size= 0.75
     , random_state=4333
     )

x_train, x_val, y_train, y_val =train_test_split(
    x_train,y_train,
    train_size = 0.5
)


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train) #Train을 보고 기준을 정해!⭐⭐⭐
x_train = scaler.transform(x_train) # ⭐ Train이 기준을 정함
x_test = scaler.transform(x_test)   # ⭐ Test는 그 기준을 사용

print(np.min(x_train),np.max(x_train)) # 0.0 1.0000000000000004
print(np.min(x_test),np.max(x_test))  #-0.0012367054167697258 1.7722477348889163

# exit()
#2. 모델 구성 
model = Sequential()
model.add(Dense(7,activation='relu',input_dim=8))
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
# print(results)

y_predict = model.predict(x_test)
from sklearn.metrics import r2_score , mean_squared_error
r2 =  r2_score(y_test, y_predict)
print ("r2 = :" , r2)

#loss(mse) :34.15455627441406
# r2 = : 0.589704701049621 (0.75이상 )

mse = mean_squared_error(y_test, y_predict) #원값 과 예측값
print("mse :" , mse)

def RMSE(y_test, y_predict):         # RMSE 함수를 정의하기
    return np.sqrt(mean_squared_error(y_test, y_predict))
    
rmse  = RMSE(y_test, y_predict)
print("RMSE : ", rmse)












