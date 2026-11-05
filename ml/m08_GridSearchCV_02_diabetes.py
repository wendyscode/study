import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import time
import warnings
warnings.filterwarnings('ignore')

#1. 데이터
x, y = load_diabetes(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=333, train_size=0.8
)

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=333)

parameters = {
    'n_estimators': [100, 300],
    'max_depth': [2, 3, 5],
    'learning_rate': [0.1, 0.05, 0.01]
}

#2. 모델
xgb = XGBRegressor(random_state=333)
model = GridSearchCV(
    xgb, parameters, cv=kfold,
    verbose=1,
    n_jobs=-1,
    scoring='r2'
)

#3. 훈련
start = time.perf_counter()
model.fit(x_train, y_train)
end = time.perf_counter()

print("최적의 매개변수 : ", model.best_estimator_)
print("최적의 파라미터 : ", model.best_params_)

#4. 평가, 예측
y_pred = model.predict(x_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print('model.score (R2):', model.score(x_test, y_test))
print('best_score (R2):', model.best_score_)
print('test RMSE:', rmse)

print("걸린시간 : ", round(end - start, 2), "초")

