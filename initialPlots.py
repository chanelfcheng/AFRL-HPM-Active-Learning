import seaborn as sb
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import csv

## Load data
df = pd.read_csv('./data/optimization.csv')
n = len(df.index)
print(n)

## Histograms
fig, axs = plt.subplots(1, 2, figsize=(10,8))
axs[0].hist(df.iloc[:, 6], color='red', edgecolor='black')
axs[0].set_title('Distribution for max frequency')
axs[1].hist(df.iloc[:, 7], color='blue', edgecolor='black')
axs[1].set_title('Distribution for middle frequency')
fig.savefig('./figures/hist.png', bbox_inches='tight')

## Correlation heatmap
plt.figure(figsize=(10,8))
correlation = sb.heatmap(df.corr(), annot=True, center=0, cmap='coolwarm')
fig = correlation.get_figure()
fig.savefig('./figures/correlation_heatmap.png', bbox_inches='tight')
