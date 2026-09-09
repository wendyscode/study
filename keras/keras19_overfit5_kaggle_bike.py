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

#2. 모델구성 
model = Sequential()
model.add(Dense(64,activation = 'relu',input_dim=8))
model.add(Dense(32,activation = 'relu'))
model.add(Dense(16,activation = 'relu'))
model.add(Dense(1,activation = 'relu'))

#3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam')
hist = model.fit(x_train,y_train,epochs=1000,batch_size=32,
          validation_data = (x_val, y_val))
#4. 평가예측 
loss = model.evaluate(x_test,y_test)
print('loss :', loss)
result = model.predict(x_test)
print(result)

y_predict = model.predict(x_test)
from sklearn.metrics import r2_score, mean_squared_error
rmse = np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", rmse)

print(submission)
y_submit = model.predict(test_csv)

submission['count'] = y_submit
print("submission :" ,submission)
print(submission.shape)

submission.to_csv(path + "submit/" + "submit_0909_1005.csv")

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
plt.title('kaggle 자전거 대여량 Loss') 
plt.xlabel('epoch')
plt.ylabel('loss')
plt.grid() #격자표시 추가 
plt.show()

# 3 애포 부터 100애포 까지 

