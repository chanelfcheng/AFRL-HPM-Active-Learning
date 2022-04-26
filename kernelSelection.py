import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ExpSineSquared, RationalQuadratic, DotProduct, RBF

## Load data
var = 6    # selects either max frequency or middle frequency as the output
df = pd.read_csv('./data/optimization.csv')
X = df.iloc[:, 0:6].to_numpy()
y = df.iloc[:, var].to_numpy()
n = len(df.index)
print(n)

## Prepare models
models = []
#models.append(( 'expsinesquared', GaussianProcessRegressor(kernel=ExpSineSquared()) ))
models.append(( 'rationalquadratic', GaussianProcessRegressor(kernel=RationalQuadratic()) ))
#models.append(( 'dotproduct', GaussianProcessRegressor(kernel=DotProduct()) ))
models.append(( 'rbf', GaussianProcessRegressor(kernel=RBF(), n_restarts_optimizer=10) ))

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
fig.suptitle('Kernel comparison')
ax = fig.add_subplot(111)
colors = {'patch_artist': True,
             'boxprops': dict(color='blue', facecolor='white'),
             'capprops': dict(color='blue'),
             'medianprops': dict(color='red'),
             'whiskerprops': dict(color='blue')}
plt.boxplot(results, **colors)
ax.set_xticklabels(names)
plt.show()
