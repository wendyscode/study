import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split

# 1. 데이터
DATA_PATH = Path(r"C:\study\_data\kaggle_bike\train.csv")
df = pd.read_csv(DATA_PATH)

datetime = pd.to_datetime(df.pop("datetime"))
y = df["count"]
# count = casual + registered 이므로 두 컬럼은 누수 방지를 위해 제외
x = df.drop(columns=["count", "casual", "registered"])
x["year"] = datetime.dt.year
x["month"] = datetime.dt.month
x["day"] = datetime.dt.day
x["hour"] = datetime.dt.hour
x["dayofweek"] = datetime.dt.dayofweek
x["dayofyear"] = datetime.dt.dayofyear
print(f"x shape = {x.shape}, y shape = {y.shape}")

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, shuffle=True, random_state=333
)

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=333)

# 2. 모델
model = HistGradientBoostingRegressor(random_state=333)

# 3. 훈련, 교차검증 평가
scores = cross_val_score(
    model, x_train, y_train, cv=kfold,
    scoring="neg_root_mean_squared_error", n_jobs=-1
)
rmse_scores = -scores
print(f"fold rmse = {rmse_scores}")
print(f"평균 rmse = {rmse_scores.mean():.4f}")

model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print(f"test rmse = {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
print(f"test r2 = {r2_score(y_test, y_pred):.4f}")
