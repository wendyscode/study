from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout #💛
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
model = Sequential()
model.add(Dense(10, input_dim=13, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dropout(0.2))           #💛
model.add(Dense(10, activation='relu'))
model.add(Dropout(0.3))           #💛
model.add(Dense(10, activation='relu'))
model.add(Dropout(0.5))           #💛
model.add(Dense(1))

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint #🤎🤎🤎🤎🤎

# 3. 컴파일, 훈련
from tensorflow.keras.optimizers import Adam
learning_rate = 0.01
# learning_rate = 0.001 #디폴트
# learning_rate = 0.0001
# learning_rate = 0.005
# learning_rate = 0.05
# learning_rate = 0.009
model.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate))

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
es = EarlyStopping(
        monitor='val_loss',
        mode='auto',
        patience=40,
        verbose=1,
        restore_best_weights=True,
)

rlr = ReduceLROnPlateau(
    monitor='val_loss',
    mode= 'auto',
    patience=20,
    verbose=1,
    factor=0.5,

)

start_time = time.time() # 현재시간을 반환. 시작시간
hist = model.fit(x_train, y_train,
                callbacks=[es, rlr ], verbose=1,
                  epochs=100, batch_size=32, validation_data=(x_val, y_val))
end_time = time.time() # 현재시간을 반환. 끝시간


print("=======================================")
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
r2 = : 0.686017566852273
mse : 26.13710610282764
RMSE :  5.112446195592443
'''







