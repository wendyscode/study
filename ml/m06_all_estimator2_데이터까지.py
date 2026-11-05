import numpy as np
from sklearn.datasets import fetch_california_housing, load_diabetes
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error
from sklearn.utils import all_estimators
import sklearn as sk
import time

import warnings
warnings.filterwarnings('ignore')

# 1. 데이터
california = fetch_california_housing(return_X_y=True)
diabetes = load_diabetes(return_X_y=True)

datasets = [california, diabetes]
data_name = ['캘리포니아', '당뇨병']

# x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=333, train_size=0.8, shuffle=True)

# scaler = StandardScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)


# 2. 모델
all = all_estimators(type_filter='regressor')

n_split = 5
kfold = KFold(n_splits=n_split, shuffle=True, random_state=333)

print('scikit-learn version:', sk.__version__)
print('회귀 모델 개수:', len(all))

start = time.time()
for index, value in enumerate(datasets):
    x, y = value
    print(f'x shape = {x.shape}, y shape = {y.shape}')

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, random_state=333, train_size=0.8, shuffle=True
    )

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    print(f'========== {data_name[index]} =========== ')
    model_name = []
    model_rmse = []

    for name, model in all:
        try:
            # 2. 모델
            model = model()
            params = model.get_params(deep=False)
            if 'random_state' in params:
                model.set_params(random_state=333)

            # 3. 훈련
            scores = cross_val_score(
                model, x_train, y_train, cv=kfold,
                scoring='neg_root_mean_squared_error'
            )

            # 4. 평가
            y_pred = cross_val_predict(model, x_test, y_test, cv=kfold)
            rmse = np.round(root_mean_squared_error(y_test, y_pred), 4)
            model_name.append(name)
            model_rmse.append(rmse)

        except Exception:
            continue

    if model_rmse:
        model_name = np.array(model_name)
        model_rmse = np.array(model_rmse)
        min_index = np.argmin(model_rmse)
        print(f'최고 모델 : {model_name[min_index]}, RMSE : {model_rmse[min_index]}')
    else:
        print('평가 가능한 모델이 없습니다.')


end = time.time()
print('걸린시간 :', round(end-start, 2), '초')

