# 33-1 카피
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Conv2D,Flatten, GlobalAveragePooling2D #💛💛💛💛💛
import time

#1. 데이터 
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target
 
x_train, x_test, y_train, y_test = train_test_split(
    x,y ,train_size= 0.75
     , random_state=4333
     )

# x_train, x_val, y_train, y_val =train_test_split(
#     x_train,y_train,
#     train_size = 0.5
#     , random_state=4333
# )

print(x.shape, y.shape) #(20640,8) (20640,)
print(x_train.shape, y_train.shape) #(7740, 8) (7740,)
print(x_test.shape, y_test.shape) #(5160, 8) (5160,)

# exit()

from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler
from sklearn.preprocessing import RobustScaler
scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()

# scaler.fit(x_train) 
# x_train = scaler.transform(x_train) 
x_train = scaler.fit_transform(x_train) 
x_test = scaler.transform(x_test)   

print(np.min(x_train),np.max(x_train))
print(np.min(x_test),np.max(x_test)) 

# x_train = x_train/255.  #점찍으면 플로트(실수)표현
# x_test = x_test/255.
print(np.max(x_train), np.min(x_train)) #1.0 0.0 #0→0.0=⚫검정
print(np.max(x_test), np.min(x_test))   #1.0 0.0 #255→1.0=⚪흰색

x_train = x_train.reshape(-1,8,1,1)
x_test = x_test.reshape(-1,8,1,1)
print(x_train.shape, x_test.shape) 

# exit()
#2. 모델구성
model = Sequential()
model.add(Conv2D(64,(3,1),activation='relu',input_shape=(8,1,1)))  #26,26,64
model.add(Conv2D(filters=32, kernel_size=(3,1),activation='relu'))  #(24,24,32)
# model.add(Dropout(0.2))
model.add(Conv2D(32,(2,1,), activation='relu')) #(23,23,32)
model.add(Conv2D(16,(2,1,), activation='relu')) #(22,22,16)
model.add(Dropout(0.2))
model.add(Conv2D(32,(2,1,), activation='relu')) #(21,21,32)
# model.add(Dropout(0.2))
model.add(GlobalAveragePooling2D())#🧀
model.add(Dense(units=32,activation='relu')) #👉 뉴런 32개를 만들겠다
model.add(Dropout(0.2))
model.add(Dense(units=16,activation='relu'))
model.add(Dense(1,))   #(10,)
model.summary()
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint #🤎🤎🤎🤎🤎

#3. 컴파일 훈련
model.compile(loss='mse',optimizer = 'adam')

es = EarlyStopping(monitor='val_loss', mode= 'min',
                    patience= 30,
                    restore_best_weights= True,
                    verbose=1, #🤎
                   )

start_time = time.time()
hist = model.fit(x_train,y_train,
                 epochs=1000,batch_size=32,validation_split = 0.2,
                callbacks=[es,],      
                # callbacks=[es,mcp],           #🤎🤎🤎🤎🤎
                verbose=1,
                 )
end_time = time.time()


print("=======================================")
#4. 평가예측 
loss = model.evaluate(x_test,y_test)
print('loss : ', loss)
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
결과값
r2 = : 0.745246898244462
mse : 0.33098992809952466
RMSE :  0.5753172412673938

r2 = : 0.6987254246253787
mse : 0.39143331074002496
RMSE :  0.6256463144141624

cnn
r2 = : -0.8608132634948669
mse : 2.4176759538803876
RMSE :  1.554887762470458
'''




