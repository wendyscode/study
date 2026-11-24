import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

# 1. 데이터
x, y = load_diabetes(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=333
)

# 2. 모델
# 기본 결정트리를 여러 개 학습하는 배깅 회귀 모델
model = BaggingRegressor(
    estimator=DecisionTreeRegressor(random_state=333),
    n_estimators=100,
    max_samples=0.8,
    bootstrap=True,  # 복원 추출
    random_state=333,
    n_jobs=-1,
)

# 비교용 랜덤 포레스트를 사용하려면 위 model 대신 아래 줄을 사용하세요.
# model = RandomForestRegressor(n_estimators=100, random_state=333, n_jobs=-1)

# 3. 훈련
model.fit(x_train, y_train)

# 4. 평가 및 예측
y_pred = model.predict(x_test)
print('최종점수 (R2):', model.score(x_test, y_test))
print('r2:', r2_score(y_test, y_pred))
print('RMSE:', np.sqrt(mean_squared_error(y_test, y_pred)))

# BaggingRegressor 옵션 참고
# bootstrap=True  : 복원 추출로 표본을 구성 (배깅)
# bootstrap=False : 비복원 추출로 표본을 구성 (페이스팅)
