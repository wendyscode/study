# https://dacon.io/competitions/open/235576/overview/description
#서울시 따릉이 대여량 예측 경진대회!!

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd

#1. 데이터                        #.은현재폴더 study / 는 하위폴더  

path = "c:/study/_data/ddarung/"  #절대경로

train_csv = pd.read_csv(path + "train.csv", index_col=0)
print(train_csv)     


test_csv = pd.read_csv(path + "test.csv", index_col=0)
print(test_csv) #[715 rows x 9 columns]

submission = pd.read_csv(path + "submission.csv", index_col=0)
print(submission)

print(train_csv.shape)#(1459, 10)
print(test_csv.shape) #(715, 9)
print(submission.shape) #(715, 1)

print(train_csv.columns)
# #Index(['hour', 'hour_bef_temperature', 'hour_bef_precipitation',
#        'hour_bef_windspeed', 'hour_bef_humidity', 'hour_bef_visibility',
#        'hour_bef_ozone', 'hour_bef_pm10', 'hour_bef_pm2.5', 'count'],
#       dtype='str')
print(train_csv.info())
print(test_csv.info())

# exit()
################################겉측치 처리 1. 삭제 ###########################
train_csv = train_csv.dropna()
print(train_csv) #[1328 rows x 10 columns]

##############################train_csv를 x와 y로 분리########################## 
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

#2.모델구성
model = Sequential()
model.add(Dense(64, input_dim=9))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(1))

#3.컴파일 훈련
model.compile(loss='mse',optimizer='adam')
hist = model.fit(x_train, y_train , epochs=1000, batch_size=32,
          validation_data = (x_val, y_val))

#4. 평가예측 
loss = model.evaluate(x_test,y_test)
print('loss :', loss)
results = model.predict(x_test)
print(results)

y_predict = model.predict(x_test)
from sklearn.metrics import r2_score , mean_squared_error
rmse = np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE :", rmse)

print("===================history========================")
print(hist)
print("===================hist.history========================")
print(hist.history)
print("===================loss========================")
print(hist.history["loss"])
print("===================val_loss========================")
print(hist.history["val_loss"])
print("===========================================")

####그래프그리기####
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Malgun Gothic'       #한글깨짐현상 폰트지정해주기 

plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'][2:],c='red',label='loss')# y값만 넣으면 시간순으로 그려줌
plt.plot(hist.history['val_loss'][2:],c='blue',label='val_loss')
plt.legend(loc= 'upper right')#우측상단에 라벨표시  #location 위치어디로할건지 
plt.title('서울시 따릉이 대여량 Loss') 
plt.xlabel('epoch')
plt.ylabel('loss')
plt.grid() #격자표시 추가 
plt.show()

# 3 애포 부터 100애포 까지 