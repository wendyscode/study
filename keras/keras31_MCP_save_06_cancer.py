import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, load_model #🩷
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import time
from tensorflow.keras.callbacks import EarlyStopping


from sklearn.datasets import load_breast_cancer #유방암 관련 데이터셋 불러오기

#1. 데이터
datasets = load_breast_cancer()
print(datasets.DESCR)   #행 569 열 30
print(datasets.feature_names)

# x = datasets.data
x = datasets['data']
y = datasets.target

print(x.shape,y.shape)  #(569, 30) (569,)
print(type(x))          #<class 'numpy.ndarray'>

print(y)
#0과 1의 개수가 몇개인지 찾아보기. - numpy 
print(np.unique(y))     #[0 1]#y 안에 있는 값들을 중복 없이 한 번씩만 보여줘!
print(np.unique(y, return_counts= True)) #(array([0, 1]), array([212, 357])) #뭐가 있는지 + 각각 몇 개인지 알려줘
#0과 1의 개수가 몇개인지 찾아보기. - pandas
print(pd.DataFrame(y).value_counts())
# 1    357
# 0    212
print(pd.Series(y).value_counts())
# 1    357
# 0    212

x_train, x_test, y_train, y_test = train_test_split(
    x,y,
    train_size=0.7,    
    # test_size=0.7,
    #shuffle=True,
    random_state=333,
      stratify=y,   #데이터를 나눌 때 y의 비율을 똑같이 유지하라
)
print(np.unique(y_train, return_counts=True))
# (array([0, 1]), array([148, 250]))
print(np.unique(y_test, return_counts=True))
#(array([0, 1]), array([ 64, 107]))

print(x_train.shape, x_test.shape)  #(398, 30) (171, 30)
print(y_train.shape, y_test.shape)  #(398,) (171,)

from sklearn.preprocessing import MinMaxScaler,StandardScaler, MaxAbsScaler
from sklearn.preprocessing import RobustScaler
# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()

scaler.fit(x_train) #Train을 보고 기준을 정해!⭐⭐⭐
x_train = scaler.transform(x_train) # ⭐ Train이 기준을 정함
x_test = scaler.transform(x_test)   # ⭐ Test는 그 기준을 사용

#2.모델구성
model = Sequential()
model.add(Dense(64, input_dim=30, activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint #🤎🤎🤎🤎🤎

#3. 컴파일 훈련
model.compile(loss='mse',optimizer = 'adam',metrics = ['acc'],)

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
filepath ="".join([path,"k31_cancer_",date,"-", filename])

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

#4. 평가예측 
loss = model.evaluate(x_test,y_test)
print("=======================================")
print('loss :', loss[0])
print('acc : ', round(loss[1],4))   #소수넷째자리까지 반올림
print("=======================================")

y_pred = model.predict(x_test)
# print(y_pred[:10])
y_pred = np.round(y_pred)
# print(y_pred[:10])

from sklearn.metrics import accuracy_score
acc_score = accuracy_score(y_test,y_pred)
print("acc_score :",acc_score)


'''
=======================================
loss : 0.024358807131648064
acc :  0.9649
=======================================

'''