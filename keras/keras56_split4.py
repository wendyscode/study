import numpy as np

a = np.array(range(1,101))
x_predict = np.array(range(96,106)) #101~106 찾자.

size = 6

# 데이터를 reshape한후 , split_x 함수로 시계열데이터로 변환 
# (N, 10, 1) -> (N, 5, 2)
# 결과를 뽑는다.

abc = a.reshape(-1,10)
print(" a reshape 결과 모양 ", abc.shape)


# 로스는 0.1이하
# 결과는 
# [101,102,103,104,105,106] 의 근사치가 나오면 됨

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM, SimpleRNN, GRU, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


# print(a)
def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i : (i+size)]
        aaa.append(subset)
    return np.array(aaa)

bbb = split_x(abc, size)
print(bbb)
print("2. split_x 결과 모양:", bbb.shape)

ccc = bbb.reshape(-1, 5, 2)
print("3. 최종 변환된 (N, 5, 2) 모양:", ccc.shape)

x = ccc[:, :-1,:] # 앞의 5개
y = ccc[:, -1,:]  # 마지막 1개 찾는거임

print("=========================")
print(x)
print("=========================")
print(y) 

print(x.shape)  # 
print(y.shape)  #


#2. 모델구성
model = Sequential()
model.add(LSTM(256, input_shape=(4,2)))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(2))


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=2000, batch_size=32)


#4. 평가, 예측
result = model.evaluate(x, y)
print('loss: ', result)
x_predict = np.array([[97, 98], [99, 100], [101, 102], [103, 104]]).reshape(1,4,2)
y_predict = model.predict(x_predict)
print('101~106 찾자: ', y_predict)

