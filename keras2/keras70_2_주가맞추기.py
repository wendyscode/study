# 삼성전자 주가맞추기 (시가맞추기)
# 앙상블 쓸것 
# LSTM Conv1D 쓸것 

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, LSTM, Conv1D, Flatten
from tensorflow.keras.layers import Concatenate

#1. 데이터 
path = './_data/samsung/'
col = ['시가', '고가', '저가', '종가', '거래량']

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

df = samsung.merge(amore, on='일자').merge(hynix, on='일자')
df = df.sort_values('일자').reset_index(drop=True)

x1_df = df[['삼성_' + c for c in col]]
x2_df = df[['아모레_' + c for c in col]]
x3_df = df[['하이닉스_' + c for c in col]]
y_df = df[['삼성_시가']]

n_train = int(len(df) * 0.8)
s1, s2, s3, sy = MinMaxScaler(), MinMaxScaler(), MinMaxScaler(), MinMaxScaler()
x1_all = s1.fit(x1_df[:n_train]).transform(x1_df)
x2_all = s2.fit(x2_df[:n_train]).transform(x2_df)
x3_all = s3.fit(x3_df[:n_train]).transform(x3_df)
y_all = sy.fit(y_df[:n_train]).transform(y_df)

timesteps = 5

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

#모델구성
model = Model(inputs=[input1, input21, input31], outputs=last_output)

#3. 컴파일 , 훈련 
model.load_weights('./_data/samsung/samsung_weight.weights.h5')# 저장한 가중치 불러오기

#4. 평가, 예측
x1_last = x1_all[-timesteps:].reshape(1, timesteps, 5)
x2_last = x2_all[-timesteps:].reshape(1, timesteps, 5)
x3_last = x3_all[-timesteps:].reshape(1, timesteps, 5)
next_open = sy.inverse_transform(model.predict([x1_last, x2_last, x3_last]))
print('다음 거래일 삼성전자 시가 예측 : ', round(next_open[0][0]), '원')

'''
다음 거래일 삼성전자 시가 예측 :   224,542 원
'''