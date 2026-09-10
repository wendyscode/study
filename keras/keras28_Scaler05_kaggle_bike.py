# 파일의 경로
# https://www.kaggle.com/competitions/bike-sharing-demand/data

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

#1. 데이터
path = './_data/kaggle_bike/'

train_csv = pd.read_csv(path + 'train.csv',index_col=0)
print(train_csv) # [10886 rows x 11 columns]
test_csv = pd.read_csv(path + 'test.csv',index_col=0)
print(test_csv)  # [6493 rows x 8 columns]
submission = pd.read_csv(path + "sampleSubmission.csv",index_col=0)
print(submission)   #[6493 rows x 1 columns]

print(train_csv.shape)  #(10886, 11)
print(test_csv.shape)   #(6493, 8)
print(submission.shape) #(6493, 1)

print(train_csv.info())
print(test_csv.info())

print(train_csv.describe()) # [8 rows x 11 columns] #describe요약해서보여줘
#########################결측치 확인 ######################
print(train_csv.isna().sum()) # 
print(test_csv.isnull().sum()) #결측치 확인 isna = isnull

########################### x,y 분리#####################
x = train_csv.drop(['casual','registered','count'], axis=1)
print(x)     #[10886 rows x 8 columns]
y = train_csv['count']
print(y, y.shape)   # (10886,)

x_train, x_test, y_train, y_test = train_test_split(
    x,y,
    train_size=0.8,    
    #test_size=0.2
    #shuffle=True
    random_state=333)
print(x.shape,y.shape)

x_train, x_val , y_train, y_val = train_test_split(
                                    x_train, y_train,
                                   train_size = 0.5,
                                   random_state = 123,)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train) #Train을 보고 기준을 정해!⭐⭐⭐
x_train = scaler.transform(x_train) # ⭐ Train이 기준을 정함
x_test = scaler.transform(x_test)   # ⭐ Test는 그 기준을 사용
x_val = scaler.transform(x_val)

#2. 모델구성 
model = Sequential()
model.add(Dense(64,activation = 'relu',input_dim=8))
model.add(Dense(32,activation = 'relu'))
model.add(Dense(16,activation = 'relu'))
model.add(Dense(1))

#3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam')
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(
    monitor='val_loss',
    mode = 'min',
    patience= 20,
    restore_best_weights= True,
)
hist = model.fit(x_train,y_train,epochs=1000,batch_size=32,
          validation_data = (x_val, y_val),callbacks=[es],)
 
#4. 평가예측 
loss = model.evaluate(x_test,y_test) 
print('loss :' , loss)
 
y_predict = model.predict(x_test)
from sklearn.metrics import r2_score , mean_squared_error
r2 =  r2_score(y_test, y_predict)
print ("r2 = :" , r2)
mse = mean_squared_error(y_test, y_predict) #원값 과 예측값
print("mse :" , mse)

def RMSE(y_test, y_predict):         # RMSE 함수를 정의하기
    return np.sqrt(mean_squared_error(y_test, y_predict))
    
rmse  = RMSE(y_test, y_predict)
print("RMSE : ", rmse)

'''
성능비교 
기존 : 
loss : 66519.640625
69/69 ━━━━━━━━━━━━━━━━━━━━ 0s 842us/step
r2 = : -1.1284406185150146
mse : 66519.625
RMSE :  257.9139876005177

Minmax : 
loss : 21280.310546875
69/69 ━━━━━━━━━━━━━━━━━━━━ 0s 743us/step
r2 = : 0.31909018754959106
mse : 21280.306640625
RMSE :  145.87771125372444

'''

