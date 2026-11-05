import numpy as np
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_digits
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict, StratifiedKFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error, accuracy_score
from sklearn.datasets import fetch_california_housing
from sklearn.utils import all_estimators
import sklearn as sk
import time

import warnings
warnings.filterwarnings('ignore')

# 1. 데이터
iris = load_iris(return_X_y=True)
cancer = load_breast_cancer(return_X_y=True)
wine = load_wine(return_X_y=True)
digits = load_digits(return_X_y=True)

datasets = [iris, cancer, wine, digits]
data_name = ['아이리스', '캔서', '와인', '디지트']

# x_train, x_test, y_train, y_test = train_test_split(x,y, stratify=y, random_state=333,train_size=0.8, shuffle=True)

# scaler = StandardScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)


# 2. 모델
all = all_estimators(type_filter='classifier')

n_split = 5
kfold = StratifiedKFold(n_splits=n_split, shuffle=True, random_state=333)

start = time.time()
for index, value in enumerate(datasets):
    x, y = value
    x_train, x_test, y_train, y_test = train_test_split(x,y, stratify=y, random_state=333,train_size=0.8, shuffle=True)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    print(f'========== {data_name[index]} =========== ')
    model_name = []
    model_acc = []

    for name, model in all:
        try:
            # 2. 모델
            model = model()

            # 3. 훈련
            scores = cross_val_score(model, x_train, y_train, cv=kfold)

            # 4. 평가
            y_pred = cross_val_predict(model, x_test, y_test, cv=kfold)

            acc = np.round(accuracy_score(y_test, y_pred),4)
            model_name.append(name)
            model_acc.append(acc)

        except:
            # print(name, '모델 오류')
            continue

    model_name = np.array(model_name)
    model_acc = np.array(model_acc)
    max_index = np.argmax(model_acc)
    print(f'최고 모델 : {model_name[max_index]}, ACC : {model_acc[max_index]}')


end = time.time()
print('걸린시간 :', round(end-start, 2), '초')

exit()

model = DecisionTreeRegressor()

# 3. 훈련, 평가
scores = cross_val_score(model, x_train, y_train, cv=kfold)
print(f'rmse = {scores}, \n 평균 rmse = {np.mean(scores):.4f}')

# acc = [1.         0.95833333 0.95833333 1.         0.95833333], 
#  평균 acc = 0.9750

y_predict = cross_val_predict(model, x_test, y_test, cv=kfold)
print(y_predict)
print(y_test)

# accuracy_score
rmse = root_mean_squared_error(y_test, y_predict)
print(rmse)