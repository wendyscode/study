import numpy as np

a = np.array(range(1,101))
x_predict = np.array(range(96,106)) #101~106 찾자.

size = 6

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

bbb = split_x(a, size)
print(bbb)
print(bbb.shape)    #(95, 6)

x = bbb[:, :-1] # 앞의 5개
y = bbb[:, -1]  # 마지막 1개 찾는거임

print("=========================")
print(x)
print("=========================")
print(y) 

print(x.shape)  # 
print(y.shape)  #


#2. 모델구성
model = Sequential()
model.add(LSTM(256, input_shape=(5,1)))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=200, batch_size=1)


#4. 평가, 예측
result = model.evaluate(x, y)
print('loss: ', result)
x_predict = np.array([[96,97,98,99,100]]).reshape(1,5,1)
y_predict = model.predict(x_predict)
print('101~106 찾자: ', y_predict)


'''
loss:  26.94070816040039
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 89ms/step
101~106 찾자:  [[104.95421]]
'''