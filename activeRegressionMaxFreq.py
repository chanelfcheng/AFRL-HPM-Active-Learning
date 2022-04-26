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
X = df.iloc[:, 0:6].to_numpy()
y1 = df.iloc[:, 6].to_numpy().reshape(-1,1)
n = len(df.index)
print(n)

## Visualize data plot
with plt.style.context('seaborn-white'):
    fig, axs = plt.subplots(figsize=(7, 7))
    tsne = TSNE(n_components=1).fit_transform(X)
    axs.scatter(x=tsne[:, 0], y=y1, c=y1, cmap='viridis', s=50)
    axs.set_title('Dispersion dataset for max frequency')
    fig.savefig('./figures/visual_plot.png', bbox_inches='tight')

## Select base kernels and weights for each regressor
kernels = []
kernels.append(RationalQuadratic())
kernels.append(RationalQuadratic())
kernels.append(RationalQuadratic())

weights = [0.1, 1, 0.1]

## Set number of initial points, committee members, and queries
n_initial = 30
n_members = len(kernels)
n_queries = 30
np.random.seed()

## Initialize committee
learner_list = list()

## Build active learners for max frequency
for member_idx in range(n_members):
    # pick intial training points
    initial_idx = np.random.choice(range(X.shape[0]), size=n_initial, replace=False)
    X_train, y1_train = X[initial_idx], y1[initial_idx]

    # remove labeled instance from data pool
    X_pool, y1_pool = np.delete(X, initial_idx, axis=0), np.delete(y1, initial_idx)

    # initialize learners
    learner = ActiveLearner(
        estimator=GaussianProcessRegressor(kernel=kernels[member_idx], alpha=1e-8),
        X_training=X_train, y_training=y1_train.reshape(-1,1)
    )
    learner_list.append(learner)

## Define query strategy
def ensemble_regression_std(regressor, X):
    _, std = regressor.predict(X, return_std=True)
    query_idx = np.argmax(std)
    return query_idx, X[query_idx]

## Assemble committee
committee = CommitteeRegressor(learner_list=learner_list, query_strategy=ensemble_regression_std)

## Visualize initial predictions by each learner
with plt.style.context('seaborn-white'):
    plt.figure(figsize=(n_members*7, 7))
    for learner_idx, learner in enumerate(committee):
        plt.subplot(1, n_members, learner_idx + 1)
        plt.scatter(x=tsne[:, 0], y=y1, c=learner.predict(X), cmap='viridis', s=50)
        plt.title('Learner no. %d initial predictions' % (learner_idx + 1))
    plt.savefig('./figures/learner_initial_preds.png', bbox_inches='tight')
    
## Visualize initial predictions by entire committee
with plt.style.context('seaborn-white'):
    plt.figure(figsize=(7, 7))
    prediction = committee.predict(X)
    plt.scatter(x=tsne[:, 0], y=y1, c=prediction, cmap='viridis', s=50)
    plt.title('Committee initial predictions')
    plt.savefig('./figures/committee_initial_preds.png', bbox_inches='tight')

## Query by committee
for idx in range(n_queries):
    query_idx, query_instance = committee.query(X_pool)
    committee.teach(X=X_pool[query_idx].reshape(1,-1), y=y1_pool[query_idx].reshape(-1,1))
    
    # remove queried instance from data pool
    X_pool = np.delete(X_pool, query_idx, axis=0)
    y1_pool = np.delete(y1_pool, query_idx)

## Visualize final predictions by each learner
preds = pd.DataFrame()
with plt.style.context('seaborn-white'):
    plt.figure(figsize=(n_members*7, 7))
    for learner_idx, learner in enumerate(committee):
        plt.subplot(1, n_members, learner_idx + 1)
        plt.scatter(x=tsne[:, 0], y=y1, c=learner.predict(X), cmap='viridis', s=50)
        plt.title('Learner no. %d final predictions' % (learner_idx + 1))
        preds[learner_idx] = learner.predict(X)[:,0]
    plt.savefig('./figures/learner_final_preds.png', bbox_inches='tight')

## Visualize final predictions by entire committee
with plt.style.context('seaborn-white'):
    plt.figure(figsize=(7, 7))
    preds = committee.predict(X)
    plt.scatter(x=tsne[:, 0], y=y1, c=preds, cmap='viridis', s=50)
    plt.title('Committee final predictions')
    plt.savefig('./figures/committee_final_preds.png', bbox_inches='tight')

print(pd.DataFrame(y1))
print(pd.DataFrame(preds))

y1_score = r2_score(y1, preds)
print(y1_score)

#results = [n_initial, n_queries, y1_score]

#with open('/home/nel227/afrl_fall_2020/data/active_learning.csv', 'a') as f:
#    csv.writer(f).writerow(results)
