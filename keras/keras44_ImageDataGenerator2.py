# 44-1 카피
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, Dropout
from tensorflow.python.keras.layers import MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import time
from sklearn.metrics import accuracy_score

#1. 데이터 

train_datagen = ImageDataGenerator(
    rescale = 1./255,
    # horizontal_flip=  True,    #수평 뒤집기, 
    # vertical_flip= True,        #수직 뒤집기, 
    # width_shift_range= 0.1,     #평형이동,
    # height_shift_range=0.1,
    # rotation_range= 5,           #각도조절(정해진 각도만큼 이미지 회전)
    # zoom_range= 1.2,
    # shear_range= 0.7,           #좌표하나를 고정하고 다른 몇개의 좌표를 이동 (한마디로 찌부)
    # fill_mode='nearest'         
)
test_datagen = ImageDataGenerator(
    rescale=1./255,
)

path_train = './_data/image/brain/train/'
path_test = './_data/image/brain/test/'

xy_train = train_datagen.flow_from_directory(
    path_train, #경로
    target_size=(100,100),
    batch_size = 32,
    class_mode = 'binary',      #이진분류
    color_mode = 'grayscale',    #흑백 
    shuffle= True,
)

xy_test = test_datagen.flow_from_directory(
    path_test,
    target_size=(100,100),
    batch_size = 32,
    class_mode = 'binary',      #이진분류
    color_mode = 'grayscale',    #흑백 
    shuffle= False,             #test에서는 필요가없다.
)
#Found 120 images belonging to 2 classes.

# exit()

# print(xy_train)
# #<keras.preprocessing.image.DirectoryIterator object at 0x000001F0ECA43FA0>
# # print(xy_train.next())  #이터레이터의 첫번째를 보여줘!! 
# # print(xy_train.next())  #두번째 이터레이터를 출력해줘

# print(xy_train[0])
# print(xy_train[1])
# print(xy_train[2])

print(xy_train[0][0])   #첫번째 배치의 x데이터가됨
print(xy_train[0][1])   #첫번째 배치의 y데이터가됨

# print(xy_train[0][0].shape) #(10, 100, 100, 1)
# print(xy_train[0][1].shape) #(10,)

# exit()
# # print(xy_train[16][0].shape)    #여기부터에러. 이유는 160장이다! 배치는 10개니까 

# print(type(xy_train))   #<class 'keras.preprocessing.image.DirectoryIterator'>
# print(type(xy_train[0]))    #<class 'tuple'>
# print(type(xy_train[0][0]))    #<class 'numpy.ndarray'>
# print(type(xy_train[0][1]))    #<class 'numpy.ndarray'>

x_train = xy_train[0][0]
y_train = xy_train[0][1]
x_test = xy_test[0][0]
y_test = xy_test[0][1]

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

# 실습 : "맹그러봐 !! "
# acc 1.0 목표 

# x_train = x_train.reshape(-1, 100,100,1)
# x_test = x_test.reshape(-1, 100,100,1)
print(x_train.shape, x_test.shape)  #(60000, 28, 28, 1) (10000, 28, 28, 1)

# from sklearn.preprocessing import OneHotEncoder
# ohe = OneHotEncoder(sparse_output=False)
# y_train = y_train.reshape(160,1)
# y_test = y_test.reshape(-1,1)
# y_train = ohe.fit_transform(y_train)
# y_test = ohe.fit_transform(y_test)

# print(y_train.shape, y_test.shape)


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