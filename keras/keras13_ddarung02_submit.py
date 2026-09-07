# https://dacon.io/competitions/open/235576/overview/description
#서울시 따릉이 대여량 예측 경진대회!!

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd

#1. 데이터                        #.은현재폴더 study / 는 하위폴더  
# path = "./_data/ddarung/"       #상대경로
path = "c:/study/_data/ddarung/"  #절대경로
# path = "c:\study\_data\ddarung/"  #슬래쉬 역슬래시 상관없다!!  #\s인식해서에러
# path = "c:/study//_data/ddarung/"   #슬래위 2개 써도 상관없다!! 
# path = "c:\\study\\_data\\ddarung\\"    #역슬래쉬 2개도 가능하다!! 
# path = "c\\study\_data\\ddarung/"   #섞어쓰기 되기만 가급적 비권장.

train_csv = pd.read_csv(path + "train.csv", index_col=0)
print(train_csv)     
# id열 포함[1459 rows x 11 columns]
# id열 안포함[1459 rows x 10 columns] index_col=0 때문

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

x_train, x_test, y_train, y_test = train_test_split(
    x,y,
    train_size=0.8,    
    #test_size=0.2
    #shuffle=True
    random_state=333)
print(x.shape,y.shape)
#***************************************************************
###############     submit물밑작업        ##################
print(test_csv.info())
################################겉측치 처리 2. 평균값 넣기 ###########################
test_csv = test_csv.fillna(test_csv.mean())
print(test_csv.info())  #(715,9)
print(test_csv.shape)   #(715, 9)

#exit()
#*************************************************************

#2.모델구성             #행무시열우선 #열=컬럼=피처=피처=속성=어트리뷰트
model = Sequential()
model.add(Dense(30, input_dim=9))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(1))

#3.컴파일 훈련
model.compile(loss='mse',optimizer='adam')
model.fit(x_train, y_train , epochs=1200, batch_size=15)


#4. 평가예측 
loss = model.evaluate(x_test,y_test)
print('loss :', loss)
results = model.predict(x_test)
print(results)

y_predict = model.predict(x_test)
from sklearn.metrics import r2_score , mean_squared_error
rmse = np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE :", rmse)

##############submission.csv 만들기 // count컬럼에 값 넣어준다################
print(submission)
y_submit = model.predict(test_csv)

submission['count']  = y_submit         #모델.predic한 결과
print(submission)
print(submission.shape)

submission.to_csv(path + "submit/" + "submit_0904_1408.csv")

