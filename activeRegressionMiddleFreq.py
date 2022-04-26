import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import csv
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ExpSineSquared, RationalQuadratic, DotProduct
from modAL.models import ActiveLearner, CommitteeRegressor
from sklearn.metrics import r2_score

## Load data
df = pd.read_csv('./data/optimization.csv')
n = len(df.index)
print(n)

## Removing points around 3.35 GHz seems to notably improve performance for middle frequency
y2 = df.iloc[:, 6]

X = df.iloc[:, 0:6].to_numpy()
y2 = df.iloc[:, 7].to_numpy().reshape(-1,1)
n = len(df.index)
print(n)

## Visualize data plot
with plt.style.context('seaborn-white'):
    fig, axs = plt.subplots(figsize=(7, 7))
    tsne = TSNE(n_components=1).fit_transform(X)
    axs.scatter(x=tsne[:, 0], y=y2, c=y2, cmap='viridis', s=50)
    axs.set_title('Dispersion dataset for middle frequency')
    fig.savefig('./figures/visual_plot2.png', bbox_inches='tight')

## Set number of initial points and queries
n_initial = 5
n_queries = 50
np.random.seed()

## Define query strategy
def GP_regression_std(regressor, X):
    _, std = regressor.predict(X, return_std=True)
    query_idx = np.argmax(std)
    return query_idx, X[query_idx]

## Build active learner for middle frequency
# pick intial training points
initial_idx = np.random.choice(range(X.shape[0]), size=n_initial, replace=False)
X_train, y2_train = X[initial_idx], y2[initial_idx]

# remove labeled instance from data pool
X_pool, y2_pool = np.delete(X, initial_idx, axis=0), np.delete(y2, initial_idx)

# initialize learner
learner = ActiveLearner(
    estimator=GaussianProcessRegressor(kernel=RationalQuadratic()),
    query_strategy=GP_regression_std,
    X_training=X_train, y_training=y2_train.reshape(-1,1),
)

## Visualize initial predictions
with plt.style.context('seaborn-white'):
    plt.figure(figsize=(7, 7))
    plt.scatter(x=tsne[:, 0], y=y2, c=learner.predict(X), cmap='viridis', s=50)
    plt.title('Learner initial predictions')
    plt.savefig('./figures/learner_initial_preds2.png', bbox_inches='tight')

## Query from the learner
for idx in range(n_queries):
    query_idx, query_instance = learner.query(X_pool)
    learner.teach(X=X_pool[query_idx].reshape(1,-1), y=y2_pool[query_idx].reshape(-1,1))
    
    # remove queried instance from data pool
    X_pool = np.delete(X_pool, query_idx, axis=0)
    y2_pool = np.delete(y2_pool, query_idx)
    
## Visualize final predictions
preds = learner.predict(X)
with plt.style.context('seaborn-white'):
    plt.figure(figsize=(7, 7))
    plt.scatter(x=tsne[:, 0], y=y2, c=preds, cmap='viridis', s=50)
    plt.title('Learner final predictions')
    plt.savefig('./figures/learner_final_preds2.png', bbox_inches='tight')

print(pd.DataFrame(y2))
print(pd.DataFrame(preds))

y2_score = r2_score(y2, preds)
print(y2_score)

#results = [n_initial, n_queries, y1_score]

#with open('/home/nel227/afrl_fall_2020/data/active_learning.csv', 'a') as f:
#    csv.writer(f).writerow(results)
