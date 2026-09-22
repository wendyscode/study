from tensorflow.keras.preprocessing.image import load_img 
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt

path = 'c:/study/_data/image/'

img = load_img(path + '내사진.jpg', target_size=(150,150))
#<PIL.Image.Image image mode=RGB size=150x150 at 0x2289A4019C0>
print(type(img))#<class 'PIL.Image.Image'>

# plt.imshow(img)
# plt.show()

arr = img_to_array(img)
print(arr)
print(arr.shape)    #(150, 150, 3)
print(type(arr))    #<class 'numpy.ndarray'>

arr = np.expand_dims(arr, axis=0) #차원 증가 
# print(arr)
print(arr.shape)    #(1, 150, 150, 3)

# np_path = './_data/kaggle_cat_dog_npy/'
# np.save(np_path + "keras48_me.npy", arr=arr)

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


it = datagen.flow(arr, 
             batch_size=1,
)
print(it)
#<keras.preprocessing.image.NumpyArrayIterator object at 0x00000171C9413F70>

# print(it.next())      #파이썬3.10까지,
print(next(it))         #파이썬3.11이후
print(next(it).shape)   #(1, 150, 150, 3)

fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(5,5))
for i in range(5):
    # batch = it.next()
    batch = next(it)
    # print(batch.shape)
    batch = batch.reshape(150,150,3)

    ax[i].imshow(batch)
    ax[i].axis('off')

plt.show()



