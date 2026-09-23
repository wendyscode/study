
from sklearn.datasets import load_diabetes
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout #💛
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
x_train, x_val , y_train, y_val = train_test_split(x_train, y_train,
                                                   train_size = 0.5,
                                                    random_state=333,)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler
from sklearn.preprocessing import RobustScaler
# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()

scaler.fit(x_train) #Train을 보고 기준을 정해!⭐⭐⭐
x_train = scaler.transform(x_train) # ⭐ Train이 기준을 정함
x_test = scaler.transform(x_test)   # ⭐ Test는 그 기준을 사용
x_val = scaler.transform(x_val)

#2. 모델구성
model = Sequential()
model.add(Dense(64,input_dim=10))
model.add(Dropout(0.2))           #💛
model.add(Dense(32))
model.add(Dropout(0.3))           #💛
model.add(Dense(16))
model.add(Dropout(0.5))           #💛
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
'''