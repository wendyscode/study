import time
import warnings

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, RandomizedSearchCV, train_test_split
from xgboost import XGBRegressor

warnings.filterwarnings('ignore')

# 1. 데이터
x, y = load_diabetes(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=333, train_size=0.8
)

kfold = KFold(n_splits=5, shuffle=True, random_state=333)

# RandomizedSearchCV는 아래 후보 조합 중 일부를 무작위로 선택해 탐색합니다.
parameters = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [2, 3, 4, 5, 7],
    'learning_rate': [0.1, 0.05, 0.01, 0.005],
    'min_child_weight': [1, 2, 3, 5, 10]
}

# 2. 모델
xgb = XGBRegressor(random_state=333)
model = RandomizedSearchCV(
    xgb, parameters, n_iter=20, cv=kfold,
    verbose=1, n_jobs=-1, scoring='r2', random_state=333
)

# 3. 훈련
start = time.perf_counter()
model.fit(x_train, y_train)
end = time.perf_counter()

print('최적의 매개변수 :', model.best_estimator_)
print('최적의 파라미터 :', model.best_params_)

# 4. 평가 및 예측
y_pred = model.predict(x_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print('model.score (R2):', model.score(x_test, y_test))
print('best_score (R2):', model.best_score_)
print('test RMSE:', rmse)
print('걸린시간 :', round(end - start, 2), '초')
