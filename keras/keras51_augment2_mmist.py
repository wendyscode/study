# 50-2 카피

from tensorflow.keras.preprocessing.image import load_img 
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import pandas as pd
import time
import time
from sklearn.metrics import accuracy_score




(x_train , y_train),(x_test,y_test) = mnist.load_data()

################# 요기부터 증폭이닷 ####################
datagen = ImageDataGenerator(
    rescale = 1./255,
    horizontal_flip=  True,    #수평 뒤집기, (좌우반전)
    # vertical_flip= True,        #수직 뒤집기, (상하반전)
    width_shift_range= 0.1,     #평형이동,
    # height_shift_range=0.1,
    rotation_range= 15,           #각도조절(정해진 각도만큼 이미지 회전)
    # zoom_range= 1.1,
    # shear_range= 0.7,           #좌표하나를 고정하고 다른 몇개의 좌표를 이동 (한마디로 찌부)
    fill_mode='nearest'         
)

augment_size = 40000

# randidx = np.random.randint(60000, size= augment_size)  #6만개중에 4만개 랜덤뽑기
print(x_train.shape[0]) #60000
randidx = np.random.randint(x_train.shape[0], size= augment_size)  
print(randidx.shape)
print(len(randidx)) #리스트는 len으로 확인 

print(np.min(randidx), np.max(randidx))

x_augmented = x_train[randidx].copy()
y_augmented = y_train[randidx].copy()

print(x_augmented.shape, y_augmented.shape) #(40000, 28, 28) (40000,)

x_augmented = x_augmented.reshape(
    x_augmented.shape[0],
    x_augmented.shape[1],
    x_augmented.shape[2], 1)
print(x_augmented.shape)    #(40000, 28, 28, 1)

x_augmented = datagen.flow(
                    x_augmented, y_augmented,
                    batch_size = augment_size,
                    shuffle= False
).next()[0]
##변환완료##
print(x_augmented.shape)    #(40000, 28, 28) (40000,)

print(x_train.shape)
x_train = x_train.reshape(60000,28,28,1)
x_test = x_test.reshape(10000,28,28,1)

x_train = np.concatenate((x_train, x_augmented))
y_train = np.concatenate((y_train, y_augmented))
print(x_train.shape, y_train.shape)

print(np.unique(y_train, return_counts=True))

#실습 맹그러봐 
#기존스코어보다 높일것 
#1. 데이터 
# (x_train, y_train),(x_test, y_test) = fashion_mnist.load_data()
print(x_train.shape, y_train.shape) #(60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)   #(10000, 28, 28) (10000,)
print(np.max(x_train),np.min(x_train))  #255 0
print(np.max(x_test),np.min(x_test))    #255 0

##### 스케일링 1 >>>>>>>>>>>0~255 → 0~1 변경 ###########
x_train = x_train/255.  #점찍으면 플로트(실수)표현
x_test = x_test/255.
print(np.max(x_train), np.min(x_train)) #1.0 0.0 #0→0.0=⚫검정
print(np.max(x_test), np.min(x_test))   #1.0 0.0 #255→1.0=⚪흰색

x_train = x_train.reshape(-1,28,28,1)
x_test = x_test.reshape(-1,28,28,1)
print(x_train.shape, x_test.shape)  #(60000, 28, 28, 1) (10000, 28, 28, 1)


from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse_output=False)
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)
y_train = ohe.fit_transform(y_train)
y_test = ohe.fit_transform(y_test)

print(y_train.shape, y_test.shape)


#2. 모델 구성 
model = Sequential()
model.add(Conv2D(64,(3,3),input_shape=(28,28,1)))
model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu'))
model.add(Dropout(0.2))
model.add(Conv2D(32,(2,2),activation='relu'))
model.add(Conv2D(16,(2,2),activation='relu'))
model.add(Dropout(0.2))
model.add(Conv2D(32,(2,2),activation='relu'))
model.add(Dropout(0.2))
model.add(Conv2D(16,(2,2),activation='relu'))
model.add(Flatten())     

model.add(Dense(units=32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=16, input_shape = (32,), activation='relu'))
model.add(Dense(10, activation='softmax'))
model.summary()

#3.컴파일 훈련 
model.compile(loss='categorical_crossentropy', optimizer = 'adam',
            metrics = ['acc'],
            )   

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(
    monitor= 'val_loss',
    mode= 'min',
    patience= 30,
    restore_best_weights= True,
)

start_time = time.time()
model.fit(x_train,y_train, epochs=500, batch_size=128,
        verbose = 1, 
        validation_split = 0.2,
         callbacks=[es],

          )
end_time = time.time()

#4. 평가예측
print("==============model.evaluate======================")
loss = model.evaluate(x_test, y_test, verbose=1)
print('loss :', loss[0])
print('loss :', loss[1])

y_predict = model.predict(x_test)

y_predict = np.argmax(y_predict,axis=1,)#.reshape(-1,1)
y_test = np.argmax(y_test,axis=1,)#.reshape(-1,1)

acc_score = accuracy_score(y_test, y_predict)

print('accuracy_score : ', acc_score)
print('걸린시간 : ', round(end_time-start_time,2), '초')


'''
accuracy_score :  0.906
걸린시간 :  137.91 초
'''