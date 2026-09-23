# 27-1 카피

from sklearn.datasets import fetch_california_housing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, root_mean_squared_error
import numpy as np
import time

# 1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) # (20640, 8) (20640,)

"""
MinMaxScaler

(원값 - Min) / (Max - Min)

[0, 1] 사이의 값으로 변환됨

아래는 잘못된 예시임
- train 데이터를 기준으로만 min, max 계산해야함
- val, test 데이터가 min, max를 구하는 기준에 포함되지 않도록 주의!!
"""

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(x)
x = scaler.transform(x)

print(x)
print(np.min(x), np.max(x)) # 0.0 1.0000000000000002


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.5, random_state=42)
print(x_train.shape, x_val.shape, x_test.shape, y_train.shape, y_val.shape, y_test.shape) # (14448, 8) (3096, 8) (3096, 8) (14448,) (3096,) (3096,)

# 2. 모델구성
model = Sequential()
model.add(Dense(16, input_dim=8))
model.add(Dense(10))
model.add(Dense(6))
model.add(Dense(1))

# 3. 컴파일, 훈련
from tensorflow.keras.optimizers import Adam
learning_rate = 0.01
# learning_rate = 0.001 #디폴트
# learning_rate = 0.0001
# learning_rate = 0.005
# learning_rate = 0.05
# learning_rate = 0.009

model.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate))

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
es = EarlyStopping(
        monitor='val_loss',
        mode='auto',
        patience=40,
        verbose=1,
        restore_best_weights=True,
)

rlr = ReduceLROnPlateau(
    monitor='val_loss',
    mode= 'auto',
    patience=20,
    verbose=1,
    factor=0.5,

)

start_time = time.time() # 현재시간을 반환. 시작시간
hist = model.fit(x_train, y_train,
                callbacks=[es, rlr ], verbose=1,
                  epochs=100, batch_size=32, validation_data=(x_val, y_val))
end_time = time.time() # 현재시간을 반환. 끝시간

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
y_pred = model.predict(x_test)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)
print("R2 :", r2)
print("MSE :", mse)
print("RMSE :", rmse)

print("걸린시간 : ", round(end_time - start_time, 2), "초") # 2번째 자리에서 반올림

# R2 : 0.6135571604704625
# MSE : 0.5109973939160222
# RMSE : 0.7148408171866113
# 걸린시간 :  4.85 초
