import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ExpSineSquared, RationalQuadratic, DotProduct
from sklearn.metrics import r2_score

## Load data
var = 6    # selects either max frequency or middle frequency as the output
df = pd.read_csv('./data/optimization.csv')
X = df.iloc[:, 0:6].to_numpy()
y1 = df.iloc[:, var].to_numpy()
n = len(df.index)
print(n)

## Split data into train and test
X_train, X_test, y1_train, y1_test = train_test_split(X, y1, train_size=0.7, random_state=0)

## Prepare models
models = []
models.append(( 'gaussian1', GaussianProcessRegressor(kernel=ExpSineSquared()) ))
models.append(( 'gaussian2', GaussianProcessRegressor(kernel=RationalQuadratic()) ))
models.append(( 'gaussian3', GaussianProcessRegressor(kernel=DotProduct()*DotProduct()) ))

## Combine predictions
preds = pd.DataFrame()
for i, model in models:
    model.fit(X_train, y1_train)
    preds[i] = model.predict(X_test)

weights = [0.1, 1, 0.1]
preds['weighted_preds'] = (preds * weights).sum(axis=1) / sum(weights)
preds['actual_vals'] = y1_test
print( preds.head())
print(r2_score(y1_test, preds['weighted_preds']))
