# https://www.kaggle.com/datasets/stytch16/jena-climate-2009-2016
# wd 를 y로
# 2016년 12월 31일 00:10:00 ~ 2017년 1월 1일 00:00:00 기간의 wd 맞추기 (144개, 타임스텝스=144)
# 데이터에 포함되어있는 2016년12월31일 데이터는 모두 자르고 할것 (과적합 방지) (144개 자르면 됨, 완전 Drop)
######################################
#수정 : T (degC) <-이놈을 y로 잡는다.
######################################

import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense,Bidirectional, Dropout, Input, LSTM, SimpleRNN, GRU
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"  #메모리 모으기

#1. 데이터
path = "./_data/kaggle_jena/"

datasets = pd.read_csv(path + 'jena_climate_2009_2016.csv', index_col=0)
size = 144

def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i : (i+size)]
        aaa.append(subset)
    return np.array(aaa)

bbb = split_x(datasets, size)
print(bbb)
print(bbb.shape)    # (420408, 144, 14)

x = np.concatenate([
    bbb[:-144, :, :2],
    bbb[:-144, :, 3:]
], axis=2)

y = bbb[:-144, -1, 2]

print(x.shape)  # (420264, 144, 13)
print(y.shape)  # (420264,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    # train_size=0.8,
    random_state=42,
)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
scaler = MinMaxScaler()

n_train, t, f = x_train.shape     # (N_train, 144, 13)
n_test = x_test.shape[0]

x_train = x_train.reshape(-1, f)  # (N_train*144, 13)
x_test  = x_test.reshape(-1, f)   # (N_test*144, 13)

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test  = scaler.transform(x_test)

x_train = x_train.reshape(n_train, t, f)
x_test  = x_test.reshape(n_test,  t, f)

print(np.min(x_train), np.max(x_train))  # 0.0 1.0000000000000004
print(np.min(x_test),  np.max(x_test))  # 0.0 1.0000000000000004


#2. 모델구성
model = Sequential()
# model.add(LSTM(256, input_shape=(144,13)))
model.add(Bidirectional(SimpleRNN(10),input_shape=(144,13))) 
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

#3. 컴파일, 훈련
from tensorflow.keras.optimizers import Adam
#learning_rate = 0.01
learning_rate = 0.001     # 0.001 - 디폴트
# learning_rate = 0.0005
#learning_rate = 0.005
#learning_rate = 0.05

model.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate))

from tensorflow.keras.callbacks import EarlyStopping

es = EarlyStopping(
    monitor = 'val_loss',
    mode = 'min',  
    patience = 30,    
    restore_best_weights = True,
)

import time
start_time = time.time()
hist = model.fit(x_train, y_train, 
                 verbose=1, 
                 epochs=20, 
                 batch_size=5000, 
                 validation_split=0.2,
                 callbacks=[es],
                 )
end_time = time.time()

print("===================================")


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss = ', loss)
print('걸린시간 = ', round(end_time - start_time, 2), "초")

# 예측 구간 = bbb의 마지막 144개 window (= 2016-12-31 00:10 ~ 2017-01-01 00:00)
x_predict = bbb[-144:, :, :-1]      # (144, 144, 13)  wd 컬럼 제외
y_true    = bbb[-144:, -1, -1]      # (144,)          실제 wd (정답 비교용)

# 학습 때 쓴 scaler로 transform만 (3D → 2D → 3D, x_train 때랑 동일)
x_predict = x_predict.reshape(-1, f)
x_predict = scaler.transform(x_predict)
x_predict = x_predict.reshape(-1, t, f)   # (144, 144, 13)

y_predict = model.predict(x_predict)      # (144, 1)

r2   = r2_score(y_true, y_predict)
rmse = np.sqrt(mean_squared_error(y_true, y_predict))
print('R2 : ', r2)
print('RMSE : ', rmse)
print('2016년 12월 31일 00:10:00 ~ 2017년 1월 1일 00:00:00 의 예측값(144개): \n', y_predict.reshape(-1))

'''
loss =  6.322382926940918
걸린시간 =  61.78 초
5/5 [==============================] - 0s 23ms/step
R2 :  -0.19544520441303903
RMSE :  58.88278383672538
2016년 12월 31일 00:10:00 ~ 2017년 1월 1일 00:00:00 의 예측값(144개): 
 [218.43329 218.3536  218.3536  218.3492  218.25447 218.15837 218.15956
 218.14723 218.1258  218.13605 218.15567 218.16354 218.19434 218.18356
 218.1515  218.15157 218.13838 218.11528 218.10081 218.04866 217.96742
 217.90529 217.86917 217.86917 217.81656 217.81656 217.79262 217.79773
 217.79262 217.72272 217.72853 217.74513 217.799   217.86076 217.75664
 217.7001  217.59883 217.54253 217.54146 217.69455 217.73393 217.6148
 217.67276 217.61603 217.59883 217.62735 217.676   217.62123 217.69257
 217.7263  217.71321 217.73586 217.696   217.724   217.7302  217.67195
 217.75066 217.78044 217.78935 217.79504 217.82791 217.89622 217.9113
 217.97676 217.97437 218.0131  218.09683 218.12395 218.15932 218.13731
 218.1245  218.20802 218.22649 218.26158 218.34718 218.51282 218.52449
 218.67833 218.67686 218.76306 218.89043 219.02129 219.0009  218.95767
 218.98334 219.14412 219.44464 219.67078 219.92116 220.51399 221.07634
 221.25014 221.30399 221.37221 221.02394 220.7589  220.5862  220.46365
 220.33669 220.25069 220.23221 220.22313 220.20729 220.00648 219.97075
 220.05225 219.87514 219.842   219.71391 219.84055 219.97511 220.2037
 219.56334 219.56728 219.55115 219.2063  218.80711 218.49925 218.34099
 218.19868 218.24677 218.24284 218.28905 218.4705  218.65883 218.66216
 218.76178 218.92851 218.89667 218.89546 219.06474 219.1299  218.84056
 218.47508 218.37724 218.57542 218.72691 218.56152 218.80771 218.30228
 218.03528 217.8129  217.70663 217.80957]
'''