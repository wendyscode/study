# 실습 목표0.77
#배치사이즈 10000
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import time

train_datagen = ImageDataGenerator(
    rescale = 1./255,
    # horizontal_flip=  True,    
    # vertical_flip= True,       
    # width_shift_range= 0.1,     
    # height_shift_range=0.1,
    # rotation_range= 5,          
    # zoom_range= 1.2,
    # shear_range= 0.7,           
    # fill_mode='nearest'         
)
test_datagen = ImageDataGenerator(
    rescale=1./255,
)

# path_train = './_data/image/cat_dog/train_set/'
# path_test = './_data/image/cat_dog/test_set/'

path_train = 'C:/study/_data/image/cat_dog/training_set/'
path_test = 'C:/study/_data/image/cat_dog/test_set/'


xy_train = train_datagen.flow_from_directory(
    './_data/image/cat_dog/training_set',
    target_size=(100, 100),
    batch_size=10000,
    class_mode='binary', # 개/고양이 이진 분류
    shuffle=True
)

xy_test = test_datagen.flow_from_directory(
    './_data/image/cat_dog/test_set',
    target_size=(100, 100),
    batch_size=10000,
    class_mode='binary',
    shuffle=False
)

print(xy_train[0][0])   #첫번째 배치의 x데이터가됨
print(xy_train[0][1])   #첫번째 배치의 y데이터가됨

x_train = xy_train[0][0]
y_train = xy_train[0][1]
x_test = xy_test[0][0]
y_test = xy_test[0][1]

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

np_path = './_data/kaggle_cat_dog_npy/'             #🤎💛🧡 🤎💛🧡
np.save(np_path + 'keras45_01_x_train.npy' , arr = xy_train[0][0])  #또는 arr = x_train 도가능 
np.save(np_path + 'keras45_01_y_train.npy' , arr = xy_train[0][1])  #🤎💛🧡 🤎💛🧡
np.save(np_path + 'keras45_01_x_test.npy' , arr = xy_train[0][0])  #🤎💛🧡 🤎💛🧡
np.save(np_path + 'keras45_01_y_test.npy' , arr = xy_train[0][1])  #🤎💛🧡 🤎💛🧡


# 2. 모델 구성 (CNN)
model = Sequential([
    Conv2D(32, (3, 3), input_shape=(100, 100, 3), activation='relu'),
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

es = EarlyStopping(monitor='val_loss', patience=100, mode='min',
                    restore_best_weights=True,
                    verbose=1, #🤎
                    )

# 3.컴파일 훈련
# generator 데이터셋 구조에 맞는 fit 실행
start_time = time.time()
hist = model.fit(
    x_train,
    y_train,
    epochs=100,
    validation_data=(x_test, y_test),
    callbacks=[es],
    verbose=1, #🤎
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
accuracy_score :  0.8116658428077114
걸린시간 :  230.22 초

accuracy_score :  0.8126544735541276
걸린시간 :  52.2 초
'''