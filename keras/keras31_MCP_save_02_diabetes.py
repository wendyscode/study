
from sklearn.datasets import load_diabetes
from tensorflow.keras.models import Sequential
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

#2. 모델구성
model = Sequential()
model.add(Dense(64,input_dim=10))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(1))
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint #🤎🤎🤎🤎🤎

#3. 컴파일 훈련
model.compile(loss='mse',optimizer = 'adam')

es = EarlyStopping(monitor='val_loss', mode= 'min',
                    patience= 30,
                    restore_best_weights= True,
                    verbose=1,
                   )
################## mcp 세이브 파일명 만들기 시작 #####################🩷🩷🩷🩷🩷
import datetime
date = datetime.datetime.now()  #현재시간반환
print(date)         #2026-09-14 11:41:15.177740
print(type(date))   #<class 'datetime.datetime'>
date = date.strftime("%m%d_%H%M")
print(date)         #0914_1147
print(type(date))   #<class 'str'> : 문자형태

path = './_save/keras31/'
filename ='{epoch:04d}-{val_loss:.4f}.keras'
filepath ="".join([path,"k31_02_diabetes_",date,"-", filename])

# 내가 생각하는 파일명 예.
# './_save/keras30/' + "k30_" + "0914_1147" + '530-0.001.keras'
#################### mcp 세이브 파일명 만들기 끝 #####################🩷🩷🩷🩷🩷
# exit()

mcp = ModelCheckpoint(                      
    monitor='val_loss', 
    mode='auto', 
    save_best_only= True, 
    filepath=  filepath,   
    verbose=1,
)                                           

start_time = time.time()
hist = model.fit(x_train,y_train,
                 epochs=1000,batch_size=32,validation_split = 0.2,
                callbacks=[es,mcp],          
                verbose=1,
                 )
end_time = time.time()


print("=======================================")


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


'''
r2 :  0.4894714687693661
mse : 2672.705147881228
RMSE :  51.6982122309972

'''