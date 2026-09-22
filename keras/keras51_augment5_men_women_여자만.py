#실습 keras51_augment5_men_women_여자만
# 여자 데이터를 증폭해서 성능을 올려봐 ! 


from tensorflow.keras.preprocessing.image import load_img 
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
import pandas as pd
import time
from sklearn.metrics import accuracy_score

#1. 데이터 

x_data = np.load('./_save/keras46/manwoman_x.npy')
y_data = np.load('./_save/keras46/manwoman_y.npy')

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x_data,
    y_data,
    test_size=0.2,
    random_state=42,
    stratify=y_data
)

################# 요기부터 증폭이닷 ####################
# [★해결] 차원 붕괴를 막기 위해 y_train 뒤에 .flatten()만 추가해주면 완벽하게 해결
x_train_women = x_train[np.where(y_train.flatten() > 0.0)]
y_train_women = y_train[np.where(y_train.flatten() > 0.0)]

print(x_train_women.shape, y_train_women.shape)
print(np.unique(y_train_women, return_counts=True))


train_datagen = ImageDataGenerator(
    # rescale = 1./255, 
    horizontal_flip=True,       # 수평 뒤집기 (좌우반전)
    width_shift_range=0.1,      # 평행이동
    rotation_range=15,          # 각도조절
    fill_mode='nearest'         
)

augment_size = 40000

# 추출된 여자 데이터 전체 개수(x_train_women.shape[0]) 중에서 40,000개를 랜덤
print("원본 여자 데이터 개수:", x_train_women.shape[0])
randidx = np.random.randint(x_train_women.shape[0], size=augment_size)  
print(randidx.shape)
print(len(randidx)) 

print(np.min(randidx), np.max(randidx))

# 위에서 정의한 x_train_women에서 랜덤 인덱스 분할
x_augmented = x_train_women[randidx].copy()
y_augmented = y_train_women[randidx].copy()

# 데이터 모양 확인: 정상적으로 (40000, 64, 64, 3)으로 추출됨을 확인
print(x_augmented.shape, y_augmented.shape) 

# [★해결] 이미 (40000, 64, 64, 3)으로 정상 형태이므로, 
# 50-2 원본 카피의 형식을 맞추기 위해 자기 자신의 차원 크기를 그대로 대입
x_augmented = x_augmented.reshape(
    x_augmented.shape[0],
    x_augmented.shape[1],
    x_augmented.shape[2],3)

print(x_augmented.shape)    

# datagen.flow를 통해 여자 데이터만 증폭 수행
x_augmented = train_datagen.flow(
                    x_augmented, y_augmented,
                    batch_size=augment_size,
                    shuffle=False
).next()[0]
##변환완료##
print(x_augmented.shape)    

print(x_train.shape)
x_train = x_train.reshape(-1, 64, 64, 3)
x_test = x_test.reshape(-1, 64, 64, 3)

# 원래 전체 데이터에 증폭 완료된 여자 데이터를 결합
x_train = np.concatenate((x_train, x_augmented))
y_train = np.concatenate((y_train, y_augmented.reshape(-1,))) # 차원 맞추기용 리셰이프 포함
print(x_train.shape, y_train.shape)

print(np.unique(y_train, return_counts=True))

# 1. 데이터 확인 및 스케일링
print(x_train.shape, y_train.shape) 
print(x_test.shape, y_test.shape)   
print(np.max(x_train), np.min(x_train))  
print(np.max(x_test), np.min(x_test))    

##### 스케일링 1 >>>>>>>>>>> 0~255 → 0~1 변경 ###########
x_train = x_train / 255.  
x_test = x_test / 255.
print(np.max(x_train), np.min(x_train)) 
print(np.max(x_test), np.min(x_test))   

x_train = x_train.reshape(-1, 64, 64, 3)
x_test = x_test.reshape(-1, 64, 64, 3)
print(x_train.shape, x_test.shape)  

y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

print(y_train.shape, y_test.shape)


# 2. 모델 구성
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid') # 이진 분류 출력층
])
model.summary()

# 3. 컴파일 훈련 
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])   

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=30,
    restore_best_weights=True,
)

start_time = time.time()
model.fit(x_train, y_train, epochs=500, batch_size=128,
          verbose=1, 
          validation_split=0.2,
          callbacks=[es])
end_time = time.time()


# 4. 평가예측
print("==============model.evaluate======================")
loss = model.evaluate(x_test, y_test, verbose=1)
print('loss :', loss[0])
print('acc :', loss[1])

y_predict = model.predict(x_test)
y_predict = np.round(y_predict)

acc_score = accuracy_score(y_test, y_predict)

print('accuracy_score : ', acc_score)
print('걸린시간 : ', round(end_time - start_time, 2), '초')

'''
accuracy_score :  0.51
걸린시간 :  132.45 초

accuracy_score :  0.8426573426573427
걸린시간 :  336.94 초
'''
