import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

# 1. 데이터
DATA_PATH = Path(r"C:\study\_data\kaggle_santander\train.csv")
df = pd.read_csv(DATA_PATH)
# ID_code는 식별자이므로 학습 특징에서 제외
x = df.drop(columns=["ID_code", "target"])
y = df["target"]
print(f"x shape = {x.shape}, y shape = {y.shape}")
print(f"target 비율 =\n{y.value_counts(normalize=True).sort_index()}")

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, shuffle=True, random_state=333, stratify=y
)

n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=333)

# 2. 모델
model = HistGradientBoostingClassifier(random_state=333)

# 3. 훈련, 교차검증 평가 (클래스 비율을 유지하고 ROC AUC로 평가)
scores = cross_val_score(
    model, x_train, y_train, cv=kfold, scoring="roc_auc", n_jobs=-1
)
print(f"fold roc_auc = {scores}")
print(f"평균 roc_auc = {scores.mean():.4f}")

model.fit(x_train, y_train)
y_score = model.predict_proba(x_test)[:, 1]
print(f"test roc_auc = {roc_auc_score(y_test, y_score):.4f}")
