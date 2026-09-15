from tensorflow.keras.models import Sequential, load_model #🩷
from tensorflow.keras.layers import Dense
from tensorflow.keras.datasets import boston_housing
import numpy as np
from sklearn.model_selection import train_test_split
import time

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

from sklearn.preprocessing import MinMaxScaler,StandardScaler, MaxAbsScaler
from sklearn.preprocessing import RobustScaler
# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()

scaler.fit(x_train) #Train을 보고 기준을 정해!⭐⭐⭐
x_train = scaler.transform(x_train) # ⭐ Train이 기준을 정함
x_test = scaler.transform(x_test)   # ⭐ Test는 그 기준을 사용
x_val = scaler.transform(x_val)

#2. 모델 구성 
path = './_save/keras31/'#🩷🩷🩷
model = load_model(path + '파일명.keras')#🩷🩷🩷

#3. 컴파일 훈련
model.save(path + '파일명.keras') #🩷🩷🩷

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


'''

'''







