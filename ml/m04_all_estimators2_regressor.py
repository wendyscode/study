import warnings
import numpy as np
import sklearn as sk
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import all_estimators

warnings.filterwarnings("ignore")

# 1. 데이터
# 여러 회귀 모델을 빠르게 비교할 수 있도록 작은 Diabetes 데이터셋 사용
x, y = load_diabetes(return_X_y=True)
print(f"x shape = {x.shape}, y shape = {y.shape}")

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, shuffle=True, random_state=123
)

# 스케일러는 훈련 데이터에만 fit하여 데이터 누수를 방지
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 2. 모든 회귀 모델 불러오기
estimators = all_estimators(type_filter="regressor")
print("scikit-learn version:", sk.__version__)
print("회귀 모델 개수:", len(estimators))

# 3. 모델별 훈련 및 평가
results = []
for name, estimator_class in estimators:
    try:
        model = estimator_class()

        # random_state 파라미터가 있는 모델은 결과를 재현 가능하게 설정
        params = model.get_params(deep=False)
        if "random_state" in params:
            model.set_params(random_state=123)

        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        results.append((name, r2, rmse))
        print(f"{name:35s} R2: {r2:8.4f} | RMSE: {rmse:8.4f}")
    except Exception as e:
        print(f"{name:35s} 모델 오류: {type(e).__name__}: {e}")

# R2 기준 상위 모델 출력
if results:
    best_name, best_r2, best_rmse = max(results, key=lambda row: row[1])
    print("\nR2 기준 최고 모델")
    print(f"{best_name}: R2 = {best_r2:.4f}, RMSE = {best_rmse:.4f}")
