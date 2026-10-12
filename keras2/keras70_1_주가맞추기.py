# 삼성전자 주가맞추기 (시가맞추기)
# 앙상블 쓸것 
# LSTM Conv1D 쓸것 

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, LSTM, Conv1D, Flatten, Dropout
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

import random
import tensorflow as tf

seed = 42
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

#1. 데이터 
path = './_data/samsung/'

col = ['시가', '고가', '저가', '종가', '거래량']     # 종목마다 5개

def load(name, pre):
    df = pd.read_csv(path + name, encoding='cp949')
    df['일자'] = pd.to_datetime(df['일자'])
    for c in col:
        df[c] = df[c].astype(str).str.replace(',', '').astype(float)
    df = df[['일자'] + col].sort_values('일자')
    df.columns = ['일자'] + [pre + c for c in col]
    return df

samsung = load('삼성전자261012.csv', '삼성_')
amore = load('아모레261012.csv', '아모레_')
hynix = load('하이닉스261012.csv', '하이닉스_')

df = samsung.merge(amore, on='일자').merge(hynix, on='일자')# 날짜가 같은 날만 합치기
df = df.sort_values('일자').reset_index(drop=True)
print(df.shape)

x1_df = df[['삼성_' + c for c in col]]
x2_df = df[['아모레_' + c for c in col]]
x3_df = df[['하이닉스_' + c for c in col]]
y_df = df[['삼성_시가']]

# 스케일링 (앞쪽 80퍼센트 기준으로만 맞춤)
n_train = int(len(df) * 0.8)
s1, s2, s3, sy = MinMaxScaler(), MinMaxScaler(), MinMaxScaler(), MinMaxScaler()
x1_all = s1.fit(x1_df[:n_train]).transform(x1_df)
x2_all = s2.fit(x2_df[:n_train]).transform(x2_df)
x3_all = s3.fit(x3_df[:n_train]).transform(x3_df)
y_all = sy.fit(y_df[:n_train]).transform(y_df)

timesteps = 5	# 최근 5일 자료로 다음 날 시가 맞추기
def split_x(data, timesteps):
    a = []
    for i in range(len(data) - timesteps):
        a.append(data[i:i + timesteps])
    return np.array(a)

x1_data = split_x(x1_all, timesteps)      # (N, 5, 5)
x2_data = split_x(x2_all, timesteps)
x3_data = split_x(x3_all, timesteps)
y_data = y_all[timesteps:]                # 다음 날 시가
print(x1_data.shape, x2_data.shape, x3_data.shape, y_data.shape)

x1_train, x1_test, x2_train, x2_test, x3_train, x3_test, y_train, y_test = train_test_split(
    x1_data, x2_data, x3_data, y_data,
    test_size=0.2,
    shuffle=False          # 시계열이라 순서를 섞지 않음
)

#2-1 모델 (삼성전자 : LSTM )
input1 = Input(shape=(timesteps, 5))
dense1 = LSTM(64, activation='tanh', name='ss1')(input1)
dense2 = Dense(32, activation='relu', name='ss2')(dense1)
output1 = Dense(16, activation='relu', name='ss3')(dense2)

#2-2 모델 (아모레 : Conv1D)
input21 = Input(shape=(timesteps, 5))
dense21 = Conv1D(64, 2, activation='relu', name='am1')(input21)
dense22 = Conv1D(32, 2, activation='relu', name='am2')(dense21)
dense23 = Flatten(name='am3')(dense22)
dense24 = Dense(32, activation='relu', name='am4')(dense23)
output21 = Dense(16, activation='relu', name='am5')(dense24)

#2-3 모델 (하이닉스 : LSTM )
input31 = Input(shape=(timesteps, 5))
dense31 = LSTM(64, activation='tanh', name='hy1')(input31)
dense32 = Dense(32, activation='relu', name='hy2')(dense31)
output31 = Dense(16, activation='relu', name='hy3')(dense32)

#2-4 모델 합치기
merge1 = Concatenate(name='mg1')([output1, output21, output31])
merge2 = Dense(32, activation='relu', name='mge2')(merge1)
merge3 = Dense(16, activation='relu', name='mge3')(merge2)
last_output = Dense(1, name='last')(merge3)


# #2-5 분기1.
# last_dense1 = Dense(10, name = 'ld1')(merge3)
# last_dense2 = Dense(10, name = 'ld2')(last_dense1)
# last_output1 = Dense(1,name='last1')(last_dense2)

# #2-6 분기2.
# last_output2 = Dense(1,name='last2')(merge3)

#모델구성
model = Model(inputs=[input1, input21, input31], outputs=last_output)
# model.summary()

#3. 컴파일 , 훈련 
model.compile(loss='mse', optimizer='adam')

es = EarlyStopping(monitor='val_loss', mode='min',
                   patience=20, restore_best_weights=True)

model.fit([x1_train, x2_train, x3_train], y_train,
          epochs=300, batch_size=16,
          validation_split=0.2,
          callbacks=[es])

#4. 평가, 예측
result = model.evaluate([x1_test, x2_test, x3_test], y_test)
print('loss : ', result)

y_predict = model.predict([x1_test, x2_test, x3_test])
y_predict = sy.inverse_transform(y_predict)
y_real = sy.inverse_transform(y_test)

print('실제 시가 마지막 5개 :', y_real[-5:].flatten())
print('예측 시가 마지막 5개 :', y_predict[-5:].flatten())

mae = np.mean(np.abs(y_real - y_predict))	# 평균 오차 (원)
print('평균 오차 : ', round(mae), '원')

x1_last = x1_all[-timesteps:].reshape(1, timesteps, 5)	# 가장 최근 5일로 다음 거래일 시가 맞추기
x2_last = x2_all[-timesteps:].reshape(1, timesteps, 5)
x3_last = x3_all[-timesteps:].reshape(1, timesteps, 5)
next_open = sy.inverse_transform(model.predict([x1_last, x2_last, x3_last]))
print('다음 거래일 삼성전자 시가 예측 : ', round(next_open[0][0]), '원')

model.save_weights('./_save/keras70/samsung_weight.weights.h5')

'''
다음 거래일 삼성전자 시가 예측 :   224,542 원
'''