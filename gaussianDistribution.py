import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import math
from sklearn.preprocessing import StandardScaler

## Load data
df = pd.read_csv('/home/nel227/afrl_fall_2020/data/optimization.csv')
n = len(df.index)
maxFreq = df.iloc[:, 6].to_numpy()
middleFreq = df.iloc[:, 7].to_numpy()

## Remove rows where max frequency deviates more than 2.0 mad
maxFreq = df.iloc[:, 6]
maxFreq_mad = stats.median_absolute_deviation(maxFreq)
maxFreq_median = np.median(maxFreq)
maxFreq_outlier = df.index[(abs(maxFreq.values - maxFreq_median) / maxFreq_mad) >= 2.0]
df = df.drop(maxFreq_outlier)

X = df.iloc[:, 0:6].to_numpy()
maxFreq = df.iloc[:, 6].to_numpy()
middleFreq = df.iloc[:, 7].to_numpy()

## Plot Gaussian distribution for max frequency
mu = np.mean(maxFreq)
sigma = np.sqrt(np.var(maxFreq))
x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
y = stats.norm.pdf(x, mu, sigma)

fig, axs = plt.subplots()
axs.plot(x, y, color='#b22222')

px = np.linspace(mu - sigma, mu + sigma, 10)
py = stats.norm.pdf(px, mu, sigma)
axs.fill_between(px, py, color='#dc1c13', alpha=0.5)

px = np.linspace(mu - sigma, mu - 2*sigma, 10)
py = stats.norm.pdf(px, mu, sigma)
axs.fill_between(px, py, color='#ea4c46', alpha=0.5)

px = np.linspace(mu + sigma, mu + 2*sigma, 10)
py = stats.norm.pdf(px, mu, sigma)
axs.fill_between(px, py, color='#ea4c46', alpha=0.5)

px = np.linspace(mu - 2*sigma, mu - 3*sigma, 10)
py = stats.norm.pdf(px, mu, sigma)
axs.fill_between(px, py, color='#f07470', alpha=0.5)

px = np.linspace(mu + 2*sigma, mu + 3*sigma, 10)
py = stats.norm.pdf(px, mu, sigma)
axs.fill_between(px, py, color='#f07470', alpha=0.5)

## Plot Gaussian distribution for middle frequency
mu = np.mean(middleFreq)
sigma = np.sqrt(np.var(middleFreq))
x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
y = stats.norm.pdf(x, mu, sigma)
plt.plot(x, y, color='#00008b')

px = np.linspace(mu - sigma, mu + sigma, 10)
py = stats.norm.pdf(px, mu, sigma)
plt.fill_between(px, py, color='#0b559f', alpha=0.5)

px = np.linspace(mu - sigma, mu - 2*sigma, 10)
py = stats.norm.pdf(px, mu, sigma)
plt.fill_between(px, py, color='#2b7bba', alpha=0.5)

px = np.linspace(mu + sigma, mu + 2*sigma, 10)
py = stats.norm.pdf(px, mu, sigma)
plt.fill_between(px, py, color='#2b7bba', alpha=0.5)

px = np.linspace(mu - 2*sigma, mu - 3*sigma, 10)
py = stats.norm.pdf(px, mu, sigma)
plt.fill_between(px, py, color='#539ecd', alpha=0.5)

px = np.linspace(mu + 2*sigma, mu + 3*sigma, 10)
py = stats.norm.pdf(px, mu, sigma)
plt.fill_between(px, py, color='#539ecd', alpha=0.5)

axs.set_xlabel('Observation')
axs.set_ylabel('Probability Density')

fig.savefig('/home/nel227/afrl_fall_2020/figures/maxFreq_gaussian_plot.png', bbox_inches='tight')
