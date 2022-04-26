import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.svm import SVR

## Load data
var = 6    # selects either max frequency or middle frequency as the output
df = pd.read_csv('./data/optimization.csv')
X = df.iloc[:, 0:6]
y = df.iloc[:, var]
n = len(df.index)
print(n)

## Prepare models
models = []
models.append(( 'linear', LinearRegression() ))
models.append(( 'gaussian', GaussianProcessRegressor() ))
models.append(( 'ridge', Ridge() ))
models.append(( 'svr', SVR() ))

## Evaluate each model
results = []
names = []

for name, model in models:
    kfold = KFold(n_splits=10)
    cv_results = cross_val_score(model, X, y, cv=kfold)
    results.append(cv_results)
    names.append(name)
    output = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(output)

# Boxplot model comparison
fig = plt.figure()
fig.suptitle('Algorithm comparison')
ax = fig.add_subplot(111)
colors = {'patch_artist': True,
             'boxprops': dict(color='blue', facecolor='white'),
             'capprops': dict(color='blue'),
             'medianprops': dict(color='red'),
             'whiskerprops': dict(color='blue')}
plt.boxplot(results, **colors)
ax.set_xticklabels(names)
plt.show()
