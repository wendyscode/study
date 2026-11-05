import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.datasets import fetch_california_housing

# 1. 데이터

x, y = fetch_california_housing(return_X_y=True)
print(x.shape, y.shape)

n_splits=5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=333)

# 2. 모델
model = HistGradientBoostingRegressor()

# 3. 평가, 훈련
scores = cross_val_score(model, x, y, cv = kfold)
print(f'rmse = {scores} \n {scores.mean():.4f}')