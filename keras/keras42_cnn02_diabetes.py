
from sklearn.datasets import load_diabetes
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D,Flatten, GlobalAveragePooling2D #💛
from sklearn.model_selection import train_test_split
import numpy as np
import time


#1. 데이터 
datasets = load_diabetes()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    train_size=0.7,
                                                    random_state=333)
# x_train, x_val , y_train, y_val = train_test_split(x_train, y_train,
#                                                    train_size = 0.5,
#                                                     random_state=333,)
print(x.shape, y.shape) #(442, 10) (442,)
print(x_train.shape, y_train.shape) #(309, 10) (309,)
print(x_test.shape, y_test.shape) #(133, 10) (133,)

# exit()

from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler
from sklearn.preprocessing import RobustScaler
scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()

scaler.fit(x_train) #Train을 보고 기준을 정해!⭐⭐⭐
x_train = scaler.transform(x_train) # ⭐ Train이 기준을 정함
x_test = scaler.transform(x_test)   # ⭐ Test는 그 기준을 사용
# x_val = scaler.transform(x_val)

x_train = x_train.reshape(-1,10,1,1) #💟💟💟💟💟
x_test = x_test.reshape(-1,10,1,1) #💟💟💟💟💟
print(x_train.shape, x_test.shape) 


#2. 모델구성
model = Sequential()
model.add(Conv2D(64,(3,1),activation='relu',input_shape=(10,1,1)))  #26,26,64 #💟💟💟💟💟
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
model.add(Dense(1,))    #💟💟💟💟💟


from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint #🤎🤎🤎🤎🤎

#3. 컴파일 훈련
model.compile(loss='mse',optimizer = 'adam')

es = EarlyStopping(monitor='val_loss', mode= 'min',
                    patience= 30,
                    restore_best_weights= True,
                    verbose=1,
                   )

start_time = time.time()
hist = model.fit(x_train,y_train,
                 epochs=1000,batch_size=32,validation_split = 0.2,
                # callbacks=[es,mcp],          
                callbacks=[es],
                verbose=1,
                 )
end_time = time.time()


print("=======================================")


#4.평가예측
loss = model.evaluate(x_test,y_test)
print('loss :', loss)
# results = model.predict(x_test)
# print(results)

y_predict = model.predict(x_test)
from sklearn.metrics import r2_score,mean_squared_error
r2 = r2_score(y_test, y_predict)
print("r2 : ", r2)
mse = mean_squared_error(y_test, y_predict) #원값 과 예측값
print("mse :" , mse)

def RMSE(y_test, y_predict):         # RMSE 함수를 정의하기
    return np.sqrt(mean_squared_error(y_test, y_predict))
    
rmse  = RMSE(y_test, y_predict)
print("RMSE : ", rmse)


'''
r2 :  0.4894714687693661
mse : 2672.705147881228
RMSE :  51.6982122309972

dropout
r2 :  0.44630366708626634
mse : 2898.696054800817
RMSE :  53.83953988288549

cnn
r2 :  0.49562674955706554
mse : 2640.481188510568
RMSE :  51.385612660652086
'''