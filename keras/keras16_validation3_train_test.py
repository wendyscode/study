from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
import numpy as np

#1.데이터
x = np.array(range(1,17))
y = np.array(range(1,17))

#[실습] train_test_split로 잘라요!! 

from sklearn.model_selection import train_test_split
x_train, x_test,y_train, y_test = train_test_split(
    x,y,
    test_size=0.5,
    # train_size=8,
    # shuffle=False,
    random_state=111,
)

x_test,x_validaion,y_test,y_validaion = train_test_split(\
    x_test,y_test,
    test_size = 0.5,
    random_state=111,
)

print(x_train,y_train)
print(x_validaion, y_validaion)
print(x_test,y_test)

