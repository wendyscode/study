import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

import joblib
import os
path = os.path.join(os.path.dirname(__file__), '_save', 'joblib')
df = joblib.load(os.path.join(path, 'm22_corr_save.joblib'))

print(df)