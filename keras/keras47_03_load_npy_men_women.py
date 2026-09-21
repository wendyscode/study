# keras47_03_load_npy_men_women.py

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPool2D, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time

# 1. 데이터 로드 (미리 저장된 npy 파일)
# 저장할 때 데이터와 라벨을 묶어서 넘파이 배열로 만들었다고 가정합니다.
x_data = np.load('./_save/keras46/manwoman_x.npy')
y_data = np.load('./_save/keras46/manwoman_y.npy')

# 데이터 확인
print(x_data.shape, y_data.shape) 
# 예: (2000, 150, 150, 3) (2000,) -> 이미지 2000장, 150x150 크기, RGB 채널

# 2. 데이터 분할 (트레인, 테스트)
x_train, x_test, y_train, y_test = train_test_split(
    x_data, y_data, train_size=0.8, shuffle=True, random_state=123
    ,stratify=y_data
)

# 3. 모델 구성 (CNN)
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPool2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPool2D(2, 2),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid') # 남/여 이진 분류이므로 출력 노드는 1개, 활성화 함수는 시그모이드
])

# 4. 컴파일 및 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

es = EarlyStopping(monitor='val_loss', patience=100, mode='min', restore_best_weights=True)
start_time = time.time()

hist = model.fit(
    x_train, y_train,
    epochs=1000,
    batch_size=32,
    validation_split=0.2,
    callbacks=[es]
)
end_time = time.time()

 # 4. 평가


print("==============model.evaluate======================")
loss = model.evaluate(x_test, y_test, verbose=1)
print('loss :', loss[0])
print('loss :', loss[1])

y_predict = model.predict(x_test)
y_predict = np.round(y_predict)

acc_score = accuracy_score(y_test, y_predict)
print('accuracy_score : ', acc_score)
print('걸린시간 : ', round(end_time-start_time,2), '초')


# 가중치 저장 폴더 생성 후 저장
model.save_weights('./_save/keras46/manwoman_weights.h5')
print("--- 가중치 저장 완료 ---")