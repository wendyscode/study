import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

# 1. 데이터
x, y = fetch_california_housing(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=444
)

# 2. 모델
# 서로 다른 회귀 모델의 예측값을 평균하는 보팅 회귀 모델
dt = DecisionTreeRegressor(max_depth=12, random_state=333)
rf = RandomForestRegressor(n_estimators=100, random_state=333, n_jobs=-1)
xgb = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.05,
                   objective='reg:squarederror', random_state=333, n_jobs=-1)

model = VotingRegressor(
    estimators=[('DT', dt), ('RF', rf), ('XGB', xgb)]
)

# 3. 훈련
model.fit(x_train, y_train)

# 4. 평가 및 예측
y_pred = model.predict(x_test)
print('최종점수 (R2):', model.score(x_test, y_test))
print('r2:', r2_score(y_test, y_pred))
print('RMSE:', np.sqrt(mean_squared_error(y_test, y_pred)))
