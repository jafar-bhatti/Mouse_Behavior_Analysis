import sys
import numpy as np
import pandas as pd
import scipy as sp
from scipy.stats import norm
import matplotlib.pyplot as plt


def compute_data(animal_id):
    try:
        data = pd.read_excel("Trial_1_raw_data_AH.xlsx", sheet_name=animal_id)  # import raw data
    except ValueError:
        print('Mouse ID invalid. Try again')
        sys.exit(1)
    data = data.to_numpy()  # convert raw data to numpy
    data = data[:, [0, 3, 4, 5, 6, 7]]  # select relevant columns
    data = data[range(17)]  # select relevant rows
    sessions = data[:, 0]  # collect the number of sessions
    hit_rate = np.zeros(sessions.size)
    for i in data[:, 0]:
        x = int(i - 1)
        if data[x, 3] == 0:
            hit_rate[x] = (data[x, 2] / (data[x, 2] + data[x, 3] + 1))
        else:
            hit_rate[x] = (data[x, 2] / (data[x, 2] + data[x, 3]))
    # This loop calculates false alarm rate. When FA = 0, we make FA = 1. When CR = 0, we add 1 to N
    fa_rate = np.zeros(sessions.size)
    for i in data[:, 0]:
        x = int(i - 1)
        if data[x, 4] == 0:
            fa_rate[x] = (1 / (data[x, 4] + data[x, 5]))
        elif data[x, 4] != 0:
            fa_rate[x] = (data[x, 4] / (data[x, 4] + data[x, 5]))
        if data[x, 5] == 0:
            fa_rate[x] = (data[x, 4] / (data[x, 4] + data[x, 5] + 1))
        elif data[x, 4] != 0:
            fa_rate[x] = (data[x, 4] / (data[x, 4] + data[x, 5]))
    # calculate percent correct
    per_cor = (data[:, 2] + data[:, 5]) / data[:, 1]
    # calculate d_prime
    d_prime = norm.ppf(hit_rate) - norm.ppf(fa_rate)
    return np.column_stack((sessions, hit_rate, fa_rate, per_cor, d_prime))


# collect data for each mouse
JB31 = compute_data('JB31')
JB32 = compute_data('JB32')
JB39 = compute_data('JB39')
JB40 = compute_data('JB40')
JB41 = compute_data('JB41')

# reorganize the data
sessions = JB31[:, 0]                                                                       # collect sessions
hr = np.column_stack((JB31[:, 1], JB32[:, 1], JB39[:, 1], JB40[:, 1], JB41[:, 1]))          # generate hit rate matrix
fa = np.column_stack((JB31[:, 2], JB32[:, 2], JB39[:, 2], JB40[:, 2], JB41[:, 2]))          # generate fa rate matrix
per = np.column_stack((JB31[:, 3], JB32[:, 3], JB39[:, 3], JB40[:, 3], JB41[:, 3]))         # generate % correct matrix
d_prime = np.column_stack((JB31[:, 4], JB32[:, 4], JB39[:, 4], JB40[:, 4], JB41[:, 4]))     # generate d prime matrix

# calculate means and standard deviations
mean_hr = np.mean(hr, axis=1)
mean_fa = np.mean(fa, axis=1)
mean_per = np.mean(per, axis=1)
mean_d = np.mean(d_prime, axis=1)

sd_hr = np.std(hr, axis=1, dtype=float)
sd_fa = np.std(fa, axis=1, dtype=float)
sd_per = np.std(per, axis=1, dtype=float)
sd_d = np.std(d_prime, axis=1, dtype=float)


# Plot mean + std for hit rate, false alarm rate, and d prime
plt.figure(figsize=(10, 6))
plt.subplot(2, 2, 1)
plt.errorbar(sessions, mean_hr, sd_hr)
plt.xlim([1, 17])
plt.ylim([0, 1])
plt.ylabel('Hit Rate')
plt.subplot(2, 2, 2)
plt.errorbar(sessions, mean_fa, sd_fa)
plt.ylabel('FA rate')
plt.xlim([1, 17])
plt.ylim([0, 1])
plt.subplot(2, 2, 3)
plt.errorbar(sessions, mean_per, sd_per)
plt.ylabel('% Correct')
plt.xlim([1, 17])
plt.ylim([0, 1])
plt.subplot(2, 2, 4)
plt.errorbar(sessions, mean_d, sd_d)
plt.xlim([1, 17])
plt.ylabel("D'")
plt.xlabel('Sessions')
plt.suptitle('Cohort 1 Average Performance (n = 5)', fontsize=16)
plt.show()
