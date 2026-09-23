import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,SimpleRNN

#1.데이터
dataset = np.array([1,2,3,4,5,6,7,8,9,10])

x = np.array([[1,2,3],
              [2,3,4],
              [3,4,5],
              [4,5,6],
              [5,6,7],
              [6,7,8],
              [7,8,9]
])

y = np.array([4,5,6,7,8,9,10])  

print(x.shape, y.shape) #(7, 3) (7,)


x = x.reshape(x.shape[0],x.shape[1],1)
print(x.shape)

