from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D    #💖💖💖💖💖

model = Sequential()    #커널사이즈 : 필터-크기+1 (커널사이즈로 잘라서 행렬연산)
model.add(Conv2D(10, (3,3),input_shape=(10,10,1)))     #💖💖💖필터10개 크기2x2
model.add(Conv2D(5,(2,2)))           #크기 크기 컬러   #💖💖💖💖💖

model.summary()
#  Layer (type)                Output Shape              Param #   
# =================================================================
#  conv2d (Conv2D)             (None, 8, 8, 10)          100                                                                        
#  conv2d_1 (Conv2D)           (None, 7, 7, 5)           205                                                                    
# =================================================================
# Total params: 305
# Trainable params: 305
# Non-trainable params: 0