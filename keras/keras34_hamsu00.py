from tensorflow.keras.models import Sequential,Model    #🩶model은 함수형모델
from tensorflow.keras.layers import Dense ,Dropout,Input #🩶

#2-1. 모델구성  / Sequential형
model = Sequential()
model.add(Dense(10, input_shape=(3,)))
model.add(Dropout(0.2))
model.add(Dense(9))
model.add(Dropout(0.2))
model.add(Dense(1))

model.summary()

########################################################🩶🩶🩶
#2-2. 함수형 모델 
input1 = Input(shape=(3,))
dense1 = Dense(10, activation='relu',name='ys1')(input1)  #model.add(Dense(10, input_shape=(3,)))
drop1 = Dropout(0.2)(dense1)            #model.add(Dropout(0.2))
dense2 = Dense(9,name='ys2')(drop1)     #model.add(Dense(9))
drop2 = Dropout(0.2)(dense2)              #model.add(Dropout(0.2))
output1 = Dense(1)(drop2)               #model.add(Dense(1))
model2 = Model(inputs=input1, outputs=output1)   #어디서부터어디까지야
model2.summary()

########################################################🩶🩶🩶
