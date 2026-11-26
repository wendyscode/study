# 스태킹: 여러 모델의 예측값을 새로운 데이터셋의 컬럼으로 만들고,
# 그 데이터셋을 다시 학습해 최종 예측을 수행합니다.

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

# 1. 데이터
x, y = load_diabetes(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=333
)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 2. 기본 모델
xgb = XGBRegressor(random_state=333, objective='reg:squarederror', verbosity=0)
lg = LGBMRegressor(random_state=333, verbosity=-1)
cat = CatBoostRegressor(random_state=333, verbose=0)
rf = RandomForestRegressor(n_estimators=100, random_state=333, n_jobs=-1)
models = [xgb, lg, cat, rf]

train_list = []
test_list = []

# 3. 각 모델의 예측값으로 새 학습/테스트 데이터 생성
for model in models:
    model.fit(x_train, y_train)
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    train_list.append(y_train_pred)
    test_list.append(y_test_pred)

    score = r2_score(y_test, y_test_pred)
    print(f'{model.__class__.__name__} R2: {score:.4f}')

x_train_new = np.array(train_list).T
x_test_new = np.array(test_list).T
print('새 학습 데이터 shape:', x_train_new.shape)
print('새 테스트 데이터 shape:', x_test_new.shape)

# 4. 메타 모델 학습
model2 = CatBoostRegressor(random_state=333, verbose=0)
model2.fit(x_train_new, y_train)

# 5. 최종 평가
y_pred_new = model2.predict(x_test_new)
print('스태킹 결과 (R2):', r2_score(y_test, y_pred_new))
print('스태킹 RMSE:', np.sqrt(mean_squared_error(y_test, y_pred_new)))
