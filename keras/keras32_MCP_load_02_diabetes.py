
from sklearn.datasets import load_diabetes
from tensorflow.keras.models import Sequential, load_model #🩷
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy as np
import time


#1. 데이터 
datasets = load_diabetes()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    train_size=0.7,
                                                    random_state=333)
x_train, x_val , y_train, y_val = train_test_split(x_train, y_train,
                                                   train_size = 0.5,
                                                    random_state=333,)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler
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
model = load_model(path + 'k31_0914_1423-0139-2258.8469.keras')#🩷🩷🩷

#3. 컴파일 훈련
model.save(path + 'k31_0914_1423-0139-2258.8469.keras') #🩷🩷🩷

#4.평가예측
loss = model.evaluate(x_test,y_test)
print('loss :', loss)
# results = model.predict(x_test)
# print(results)

y_predict = model.predict(x_test)
from sklearn.metrics import r2_score,mean_squared_error
r2 = r2_score(y_test, y_predict)
print("r2 : ", r2)
mse = mean_squared_error(y_test, y_predict) #원값 과 예측값
print("mse :" , mse)

def RMSE(y_test, y_predict):         # RMSE 함수를 정의하기
    return np.sqrt(mean_squared_error(y_test, y_predict))
    
rmse  = RMSE(y_test, y_predict)
print("RMSE : ", rmse)

