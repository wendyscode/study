from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.datasets import boston_housing
import numpy as np
from sklearn.model_selection import train_test_split

#1. 데이터 
(x_train,y_train),(x_test,y_test)=boston_housing.load_data()
print(x_train.shape,x_test.shape) #(404, 13) (102, 13)
print(y_train.shape,y_test.shape) #(404,) (102,)

# x_train,y_train,x_test,y_test = train_test_split(
#                     x,y,
#                     train_size=0.7,
#                     random_state=111,
# )

x_train, x_val, y_train , y_val = train_test_split(
                  x_train, y_train,
                  train_size=0.5,
                  random_state=111,

)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train) #Train을 보고 기준을 정해!⭐⭐⭐
x_train = scaler.transform(x_train) # ⭐ Train이 기준을 정함
x_test = scaler.transform(x_test)   # ⭐ Test는 그 기준을 사용
x_val = scaler.transform(x_val)

#2. 모델 구성 
model = Sequential()
model.add(Dense(10, input_dim=13, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

#3.컴파일 훈련
model.compile(loss= 'mse' , optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(
    monitor= 'val_loss',
    mode= 'min',
    patience= 10,
    restore_best_weights= True,
)
hist = model.fit(x_train, y_train, epochs=1000, batch_size=32,
          validation_data = (x_val, y_val),
          callbacks=[es],)

print("======================================")
#4. 평가예측 
loss = model.evaluate(x_test,y_test) 
print('loss :' , loss)

#tessoflow에서 r2를 지원안하기 때문에 직접 해줘야한다 
#싸이킷런에 있다 
#원값과 예측값!! 
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
성능비교 
기존 :
 loss : 66.09056854248047
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 9ms/step 
r2 = : 0.20606067514265214
mse : 66.09056489233741
RMSE :  8.129610377646484

Minmax : 
loss : 20.150224685668945
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 14ms/step
r2 = : 0.7579373619651317
mse : 20.150225572877776
RMSE :  4.488900263191172
'''







