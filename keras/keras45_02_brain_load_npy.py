# 45-01 카피
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, Dropout
from tensorflow.python.keras.layers import MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import time
from sklearn.metrics import accuracy_score
# 1. Load the arrays created by keras45_01_brain_save_npy.py.

np_path = './_data/brain_npy/'
x_train = np.load(np_path + 'keras45_01_x_train.npy')
y_train = np.load(np_path + 'keras45_01_y_train.npy')
x_test = np.load(np_path + 'keras45_01_x_test.npy')
y_test = np.load(np_path + 'keras45_01_y_test.npy') #❤️🩷🩶❤️🩷🩶

print(x_train.shape,y_train.shape)
print(x_train.shape,y_train.shape)

# 실습 : "맹그러봐 !! "
# acc 1.0 목표 

# x_train = x_train.reshape(-1, 100,100,1)
# x_test = x_test.reshape(-1, 100,100,1)
print(x_train.shape, x_test.shape)  #(60000, 28, 28, 1) (10000, 28, 28, 1)


# 2. 모델 구성 (CNN)
model = Sequential([
    Conv2D(32, (3, 3), input_shape=(100, 100, 1), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(124, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid') # 이진 분류 마감
])

# 3. 컴파일 및 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

es = EarlyStopping(monitor='val_loss', patience=10, mode='min', restore_best_weights=True)

# 3.컴파일 훈련
# generator 데이터셋 구조에 맞는 fit 실행
start_time = time.time()
hist = model.fit(
    x_train,
    y_train,
    epochs=5000,
    validation_data=(x_test, y_test),
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

'''
accuracy_score :  1.0
걸린시간 :  3.81 초
'''