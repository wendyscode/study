import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_score

# 1. 데이터
DATA_PATH = Path(r"C:\study\_data\kaggle_bike\train.csv")
df = pd.read_csv(DATA_PATH)

# datetime을 모델이 사용할 수 있는 숫자형 특징으로 변환
# casual, registered는 count를 구성하는 값이므로 데이터 누수를 막기 위해 제외

datetime = pd.to_datetime(df.pop("datetime"))
x = df.drop(columns=["count", "casual", "registered"])
x["year"] = datetime.dt.year
x["month"] = datetime.dt.month
x["day"] = datetime.dt.day
x["hour"] = datetime.dt.hour
x["dayofweek"] = datetime.dt.dayofweek
x["dayofyear"] = datetime.dt.dayofyear
y = df["count"]

print(f"x shape = {x.shape}, y shape = {y.shape}")

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=333)

# 2. 모델
model = HistGradientBoostingRegressor(random_state=333)

# 3. 훈련, 평가 (scikit-learn의 neg_root_mean_squared_error는 음수로 반환)
scores = cross_val_score(
    model, x, y, cv=kfold, scoring="neg_root_mean_squared_error", n_jobs=-1
)
rmse_scores = -scores
print(f"rmse = {rmse_scores}")
print(f"평균 rmse = {rmse_scores.mean():.4f}")
