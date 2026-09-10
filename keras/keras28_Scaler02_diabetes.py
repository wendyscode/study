# 얼리스타핑 적용하고 더좋앗던 성능으로 갱신시키기
# - 에포 마니주고 
# - 페이션트 크게준다 

#R2 기준 0.62 이상 만들어보기 

from sklearn.datasets import load_diabetes
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy as np

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

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train) #Train을 보고 기준을 정해!⭐⭐⭐
x_train = scaler.transform(x_train) # ⭐ Train이 기준을 정함
x_test = scaler.transform(x_test)   # ⭐ Test는 그 기준을 사용
x_val = scaler.transform(x_val)

#2. 모델구성
model = Sequential()
model.add(Dense(64,input_dim=10))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(1))

#3.컴파일 훈련
model.compile(loss='mse',optimizer='adam')
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(
    monitor='val_loss',
    mode = 'min',
    patience= 30,
    restore_best_weights= True,
)

hist = model.fit(x_train, y_train, epochs=100, batch_size=2,
          validation_data = (x_val, y_val),
          callbacks = [es],)

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
성능비교 
기존 : 
loss : 2644.657470703125
5/5 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step 
r2 :  0.4948290374186849
mse : 2644.65734950521
RMSE :  51.4262321146048

Minmax : 
loss : 2610.520263671875
5/5 ━━━━━━━━━━━━━━━━━━━━ 0s 9ms/step 
r2 :  0.5013497911668592
mse : 2610.5200759843733
RMSE :  51.093248829805034
'''