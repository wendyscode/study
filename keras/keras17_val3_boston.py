from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.datasets import boston_housing
import numpy as np
from sklearn.model_selection import train_test_split

#1. 데이터 
(x_train,y_train),(x_test,y_test)=boston_housing.load_data()
print(x_train.shape,x_test.shape) #(404, 13) (102, 13)
print(y_train.shape,y_test.shape) #(404,) (102,)

# x_train,y_train,x_test,y_test = train_test_split(
#                     x,y,
#                     train_size=0.7,
#                     random_state=111,
# )

x_train, x_val, y_train , y_val = train_test_split(
                  x_train, y_train,
                  train_size=0.5,
                  random_state=111,

)
#2. 모델 구성 
model = Sequential()
model.add(Dense(10, input_dim=13))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

#3.컴파일 훈련
model.compile(loss= 'mse' , optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=2,
          validation_data = (x_val, y_val))

print("======================================")
#4. 평가예측 
loss = model.evaluate(x_test,y_test) 
print('loss :' , loss)

#tessoflow에서 r2를 지원안하기 때문에 직접 해줘야한다 
#싸이킷런에 있다 
#원값과 예측값!! 
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









