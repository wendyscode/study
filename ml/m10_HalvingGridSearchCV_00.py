# 9-0 복사

import numpy as np
from sklearn.datasets import load_digits, load_iris
from sklearn.model_selection import train_test_split, StratifiedKFold # classification 일떈 stratifiedkfold
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection._search_successive_halving import HalvingGridSearchCV
from xgboost import XGBClassifier
import time
import warnings
warnings.filterwarnings('ignore')

# 1. 데이터
x, y = load_digits(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=333, train_size=0.8, shuffle=True, stratify=y)

# print(x_train.shape)
# print(np.unique(y, return_counts=True))
# # exit()

n_splits =5
kfold = StratifiedKFold(n_splits=n_splits, shuffle = True, random_state=333)

parameters =[
    {'n_estimators' : [100,500], 'max_depth ' :[6, 10, 12],
     'learning_rate' : [0.1, 0.01, 0.001]}, #18 가지 경우의 수
     {'max_depth' : [6, 8, 10, 12], 'learning_rate':[0.1, 0.01, 0.001]},
     {'min_child_weight':[2,3,5,10], 'learning_rate':[0.1, 0.01, 0.001]}
]

# 2. 모델
xgb = XGBClassifier()
model = HalvingGridSearchCV(xgb, parameters, cv=kfold,
                    verbose=1,
                    # n_jobs=-1,
                    # random_state=100,
                    refit=True,
                    # n_iter=10,
                    min_resources=100,  # n_splits * 2 * 10 (클래스개수, classifier일 경우 클래스 개수 까지 곱함)
                    max_resources=1000 # 데이터 적을땐 이 모델은 좀 별로                    factor=2,
                    )


# 3. 훈련, 예측
print('===============================')
start = time.perf_counter()
model.fit(x_train, y_train)
end = time.perf_counter()

print('최적의 매개변수 :', model.best_estimator_)
print('최적의 파라미터 :', model.best_params_)

# 4. 평가, 예측
print(f'model.score : {model.score(x_test, y_test):.4f}')
print(f'best_score : {model.best_score_:.4f}')

print(f' 걸린 시간 : {end - start:.4f}초')