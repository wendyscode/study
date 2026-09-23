# https://dacon.io/competitions/open/235576/overview/description
#서울시 따릉이 대여량 예측 경진대회!!

import numpy as np
from tensorflow.keras.models import Sequential, load_model #🩷
from tensorflow.keras.layers import Dense, Dropout #💛
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd
import time

#1. 데이터                        #.은현재폴더 study / 는 하위폴더  

path = "c:/study/_data/ddarung/"  #절대경로

train_csv = pd.read_csv(path + "train.csv", index_col=0)
print(train_csv)     


test_csv = pd.read_csv(path + "test.csv", index_col=0)
print(test_csv) #[715 rows x 9 columns]

submission = pd.read_csv(path + "submission.csv", index_col=0)

train_csv = train_csv.dropna()
print(train_csv) #[1328 rows x 10 columns]


x = train_csv.drop(['count'], axis=1) #axis: 어느 방향으로 계산할 거냐? #열(컬럼)삭제
print(x) # [1328 rows x 9 columns]    #drop은 삭제한다는뜻! pandas에서 count를 삭제한당

y = train_csv['count']
print(y)
print(y.shape) #(1328,)

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.7,random_state=333)
print(x.shape,y.shape)

x_train,x_val , y_train, y_val = train_test_split(x_train, y_train,
                                                  train_size = 0.5,
                                                  random_state=123)

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

#2.모델구성
model = Sequential()
model.add(Dense(64, input_dim=9, activation='relu'))
model.add(Dropout(0.5))           #💛
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.3))           #💛
model.add(Dense(16, activation='relu'))
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
print('loss :', loss)
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



'''
r2 = : 0.6951428590753901
mse : 2195.5162629470456
RMSE :  46.85633642259119

'''