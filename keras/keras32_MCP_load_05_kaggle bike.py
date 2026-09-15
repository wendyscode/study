# 파일의 경로
# https://www.kaggle.com/competitions/bike-sharing-demand/data

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, load_model #🩷
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import time

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
    random_state=4333)
print(x.shape,y.shape)

x_train, x_val , y_train, y_val = train_test_split(
                                    x_train, y_train,
                                   train_size = 0.5,
                                   random_state = 123,)

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


'''

