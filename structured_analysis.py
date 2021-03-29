import numpy as np
from numpy.core.arrayprint import _leading_trailing
from numpy.core.fromnumeric import mean
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

import sys

# perc_train = int(sys.argv[1]) / 100

with open('structure_data.csv', 'r') as f:
    experiment_df = pd.read_csv(f)
experiment_df = experiment_df[experiment_df.columns.drop(['make'])]
experiment_df = experiment_df.dropna()

NUM_BOOTSTRAPS = 1000
train_frac = []
train_percentages = [10, 25, 50, 80]
perf = {perc:[] for perc in train_percentages}
fit = {perc:[] for perc in train_percentages}
coefs = {perc:[] for perc in train_percentages}

for perc_idx, perc_train in enumerate(train_percentages):
    split_idx = int(len(experiment_df)*perc_train/100)
    for idx in range(NUM_BOOTSTRAPS):
        print(f'\r Analyzing {perc_idx+1}/{len(train_percentages)} training fraction, step {idx+1}/{NUM_BOOTSTRAPS}     ', end ='')
        experiment_df = experiment_df.sample(frac=1)
        train = experiment_df[:split_idx]
        test = experiment_df[split_idx:]
        train_price, test_price = train['price'], test['price']
        train_cov, test_cov = train[train.columns.drop('price')], test[test.columns.drop('price')]

        regr = LinearRegression()
        regr.fit(train_cov, train_price)
        train_pred_price = regr.predict(train_cov)
        test_pred_price = regr.predict(test_cov)
        fit[perc_train].append(mean_squared_error(train_price, train_pred_price))
        perf[perc_train].append(mean_squared_error(test_price, test_pred_price))
        coefs[perc_train].append(regr.coef_)
# plt.figure(dpi=300)
# plt.scatter(train_price, train_pred_price, s=2, alpha=0.4)
# plt.scatter(test_price, test_pred_price, s=2, alpha=0.4)
# plt.legend(['train', 'test'])
# plt.title(f'MSE {perf}\n {perc_train}% train split')
print("\nDone")
plt.figure(dpi=300)
plt.title('test')
for key in perf:
    plt.hist(perf[key], label=f'{key}', alpha=1/(len(perf)+1))
plt.legend()

plt.figure(dpi=300)
plt.title('train')
for key in fit:
    plt.hist(fit[key], label=f'{key}', alpha=1/(len(fit)+1))
plt.legend()

# Confidence on pred error
plt.figure(dpi=300)
plt.title('Bootstrapped spread of errors against training frac')
bot, med, top, av = [], [], [], []
for key in perf:
    bot.append(np.quantile(perf[key],0.025))
    med.append(np.quantile(perf[key],0.5))
    top.append(np.quantile(perf[key],0.975))
    av.append(np.mean(perf[key]))
plt.scatter(perf.keys(), med, marker='o', c='y', label='test_median', alpha=0.5)
plt.scatter(perf.keys(), av, marker='x', c='y', label='test_average', alpha=0.5)
plt.fill_between(perf.keys(), bot, top, color='y', alpha=0.1)

bot, med, top, av = [], [], [], []
for key in fit:
    bot.append(np.quantile(fit[key],0.025))
    med.append(np.quantile(fit[key],0.5))
    top.append(np.quantile(fit[key],0.975))
    av.append(np.mean(fit[key]))
plt.scatter(fit.keys(), med, marker='o', c='b', label='train_median', alpha=0.5)
plt.scatter(fit.keys(), av, marker='x', c='b', label='train_average', alpha=0.5)
plt.fill_between(fit.keys(), bot, top, color='b', alpha=0.1)

plt.legend()

for key in coefs:
    coefs[key] = list(zip(*coefs[key]))

for idx in range(len(train_cov.columns)):
    plt.figure(dpi=300)
    plt.title(train_cov.columns[idx])
    bot, med, top, av = [], [], [], []
    for key in perf:
        bot.append(np.quantile(coefs[key][idx],0.025))
        med.append(np.quantile(coefs[key][idx],0.5))
        top.append(np.quantile(coefs[key][idx],0.975))
        av.append(np.mean(coefs[key][idx]))
    plt.scatter(perf.keys(), med, marker='o', c='y', label='median', alpha=0.5)
    plt.scatter(perf.keys(), av, marker='x', c='y', label='average', alpha=0.5)
    plt.fill_between(perf.keys(), bot, top, color='y', alpha=0.1)
    plt.legend()
plt.show()