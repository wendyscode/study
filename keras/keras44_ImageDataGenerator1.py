import numpy as np
from keras.preprocessing.image import ImageDataGenerator
print(np.__version__)

train_datagen = ImageDataGenerator(
    rescale = 1./255,
    horizontal_flip=  True,    #수평 뒤집기, 
    vertical_flip= True,        #수직 뒤집기, 
    width_shift_range= 0.1,     #평형이동,
    height_shift_range=0.1,
    rotation_range= 5,           #각도조절(정해진 각도만큼 이미지 회전)
    zoom_range= 1.2,
    shear_range= 0.7,           #좌표하나를 고정하고 다른 몇개의 좌표를 이동 (한마디로 찌부)
    fill_mode='nearest'         
)
test_datagen = ImageDataGenerator(
    rescale=1./255,
)

path_train = './_data/image/brain/train/'
path_test = './_data/image/brain/test/'

xy_train = train_datagen.flow_from_directory(
    path_train, #경로
    target_size=(100,100),
    batch_size = 10,
    class_mode = 'binary',      #이진분류
    color_mode = 'grayscale',    #흑백 
    shuffle= True,
)

xy_test = test_datagen.flow_from_directory(
    path_test,
    target_size=(100,100),
    batch_size = 10,
    class_mode = 'binary',      #이진분류
    color_mode = 'grayscale',    #흑백 
    # shuffle= True,             #test에서는 필요가없다.
)

print(xy_train)
#<keras.preprocessing.image.DirectoryIterator object at 0x000001F0ECA43FA0>
# print(xy_train.next())  #이터레이터의 첫번째를 보여줘!! 
# print(xy_train.next())  #두번째 이터레이터를 출력해줘

print(xy_train[0])
print(xy_train[1])
print(xy_train[2])

print(xy_train[0][0])   #첫번째 배치의 x데이터가됨
print(xy_train[0][1])   #첫번째 배치의 y데이터가됨

print(xy_train[0][0].shape) #(10, 100, 100, 1)
print(xy_train[0][1].shape) #(10,)

# print(xy_train[16][0].shape)    #여기부터에러. 이유는 160장이다! 배치는 10개니까 

print(type(xy_train))   #<class 'keras.preprocessing.image.DirectoryIterator'>
print(type(xy_train[0]))    #<class 'tuple'>
print(type(xy_train[0][0]))    #<class 'numpy.ndarray'>
print(type(xy_train[0][1]))    #<class 'numpy.ndarray'>

