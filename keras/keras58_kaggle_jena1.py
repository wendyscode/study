#https://www.kaggle.com/datasets/stytch16/jena-climate-2009-2016/data

# wd를 y로 하라.2016-12-31 데이터의 wd를 맞추기 
#뒤에서부터 144개를빼고 작업 컬럼
# 컬럼 전체 14개중 wd는 드랍 
# 세로로 144개는 드랍 > 서밋용 제출 2016년 12월 31일  

# x = (N,144,13)
# y = (N,144,1)

import os 
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"    #메모리모으기
import numpy as np
import pandas as pd
import time
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, GRU, SimpleRNN
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# from tensorflow.keras.layers import Dense,Conv2D,Flatten,Dropout,MaxPool2D,BatchNormalization
# from sklearn.metrics import r2_score,accuracy_score,mean_absolute_error
# from tensorflow.keras.utils import to_categorical

path='./_data/kaggle_jena/'

datasets = pd.read_csv(path +'jena_climate_2009_2016.csv',index_col=0 )

print("원본 데이터 모양:", datasets.shape)

# 마지막 144개 → 2016-12-31의 실제 wd
y_cor = datasets[-144:]['wd (deg)']
print("예측치정답데이터 :", y_cor.shape)

##########훈련할 데이터 자르기 ############
x_data = datasets[:-288].drop(['wd (deg)'], axis=1) 
y_data = datasets[144:-144][['wd (deg)']]         

print('풍향을제외한 나머지 기상데이터 :',x_data.shape) #전체 14개 컬럼 중 풍향(wd)을 제외한 나머지 13개의 기상 데이터
print('풍향(wd (deg) :',y_data.shape) #풍향(wd (deg)) 컬럼

size_x = 144
size_y = 144

def split_x (dataset,size):
    aaa = []
    for i in range(len(dataset) - size +1 ):
        subset = dataset[i : (i+size)]
        aaa.append(subset)
    return np.array(aaa)

start_time = time.time()
x = split_x(x_data,size_x)
y = split_x(y_data, size_y)
end_time = time.time()

print('데이터 자르는 데 걸린 시간 :', end_time - start_time)

# -----------------------------------

    # x_scaler = MinMaxScaler()
    # y_scaler = MinMaxScaler()

    # x_data = x_scaler.fit_transform(x_data)
    # y_data = y_scaler.fit_transform(y_data)

    # print('Scaling 후 x :', x_data.shape)
    # print('Scaling 후 y :', y_data.shape)


    # print('x :', x.shape)
    # print('y :', y.shape)

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)

print('x_train :', x_train.shape)
print('y_train :', y_train.shape)
print('x_test  :', x_test.shape)
print('y_test  :', y_test.shape)

#2. 모델구성
model = Sequential()

model.add(
    LSTM(
        64,
        input_shape=(144, 13),
        return_sequences=True
    )
)

model.add(LSTM(32, return_sequences=True))

model.add(Dense(16))
model.add(Dense(1))


model.summary()

#3. 컴파일 ,훈련
model.compile(
    loss='mse',
    optimizer='adam',
    metrics=['mae']
)
es = EarlyStopping(
    monitor='val_loss',
    patience=10,
    mode='min',
    restore_best_weights=True
)

cp = ModelCheckpoint(
    './_data/kaggle_jena/jena_best.keras',
    monitor='val_loss',
    mode='min',
    save_best_only=True
)
start = time.time()

hist = model.fit(
    x_train,
    y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.2,
    callbacks=[es, cp],
    verbose=1
)

end = time.time()

print('훈련시간 :', end - start)

#4. 평가, 예측
loss, mae = model.evaluate(
    x_test,
    y_test,
    verbose=1
)

print('loss :', loss)
print('mae  :', mae)

# 마지막 144개 데이터(2016-12-31) 예측해보기
x_pre = datasets[-288:-144].drop(['wd (deg)'], axis=1)

x_pre_scaled = x_scaler.transform(x_pre)

x_pre_3d = x_pre_scaled.reshape(1, 144, 13)

y_pred = model.predict(x_pre_3d)

print("2016-12-31 예측 wd:")
print(y_pred)


'''
# 2016-12-31 wd 예측값 저장
submission = pd.DataFrame(y_pred, columns=['wd (deg)'])

submission.to_csv(
    './_data/kaggle_jena/submission.csv',
    index=False
)

print('제출 파일 저장 완료')
print(submission)
'''
