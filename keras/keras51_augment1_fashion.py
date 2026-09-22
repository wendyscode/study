# 50-2 카피

from tensorflow.keras.preprocessing.image import load_img 
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist

(x_train , y_train),(x_test,y_test) = fashion_mnist.load_data()

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

exit()


print(x_train.shape)    #(60000, 28, 28)
print(x_train[0].shape) #(28, 28)

# aaa = np.tile(x_train[0],augment_size)
# print(aaa.shape)

aaa = np.tile(x_train[0],augment_size).reshape(-1,28,28,1)
print(aaa.shape)    #(100, 28, 28, 1)

xy_data = datagen.flow(
        np.tile(x_train[0].reshape(28*28),augment_size).reshape(-1,28,28,1),
        np.zeros(augment_size),
        batch_size = augment_size,
        shuffle= False,
    ).next()

print(xy_data)
print(type(xy_data))

# print(xy_data.shape)#AttributeError: 'tuple' object has no attribute 'shape'
print(len(xy_data)) #2가 나옴 / X와Y가 2개니까 

print(xy_data[0].shape)
print(xy_data[1].shape)

plt.figure(figsize=(7,7))
for i in range(49) : 
    plt.subplot(7,7,i+1),
    plt.imshow(xy_data[0][i], cmap='gray')
plt.show()