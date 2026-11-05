import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split

# 1. 데이터
x, y = fetch_california_housing(return_X_y=True)
print(f"x shape = {x.shape}, y shape = {y.shape}")

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, shuffle=True, random_state=333
)

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=333)

# 2. 모델
model = HistGradientBoostingRegressor(random_state=333)

# 3. 훈련, 교차검증 평가 (훈련 데이터 안에서만 K-fold 수행)
scores = cross_val_score(
    model, x_train, y_train, cv=kfold,
    scoring="neg_root_mean_squared_error", n_jobs=-1
)
rmse_scores = -scores
print(f"fold rmse = {rmse_scores}")
print(f"평균 rmse = {rmse_scores.mean():.4f}")

# 전체 훈련 데이터로 재학습한 뒤, 보류한 테스트 데이터로 최종 평가
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print(f"test rmse = {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
print(f"test r2 = {r2_score(y_test, y_pred):.4f}")
