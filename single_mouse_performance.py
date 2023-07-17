import sys
import numpy as np
import pandas as pd
import scipy as sp
from scipy.stats import norm
import matplotlib.pyplot as plt

try:
    ID = input('What is the mouse ID? ')
    data = pd.read_excel("Trial_1_raw_data_AH.xlsx", sheet_name=ID)     # import raw data
except ValueError:
    print('Mouse ID invalid. Try again')
    sys.exit(1)

data = data.to_numpy()                           # convert raw data to numpy
data = data[:, [0, 3, 4, 5, 6, 7]]               # select relevant columns
data = data[range(17)]                           # select relevant rows
sessions = data[:, 0]                            # collect the number of sessions

# This loop calculates the hit rate. When miss = 0, we add 1 to N.
hit_rate = np.zeros(sessions.size)
for i in data[:, 0]:
    x = int(i-1)
    if data[x, 3] == 0:
        hit_rate[x] = (data[x, 2] / (data[x, 2] + data[x, 3] + 1))
    else:
        hit_rate[x] = (data[x, 2]/(data[x, 2] + data[x, 3]))

# This loop calculates false alarm rate. When FA = 0, we make FA = 1. When CR = 0, we add 1 to N
fa_rate = np.zeros(sessions.size)
for i in data[:, 0]:
    x = int(i - 1)
    if data[x, 4] == 0:
        fa_rate[x] = (1 / (data[x, 4] + data[x, 5]))
    elif data[x, 4] != 0:
        fa_rate[x] = (data[x, 4]/(data[x, 4] + data[x, 5]))
    if data[x, 5] == 0:
        fa_rate[x] = (data[x, 4] / (data[x, 4] + data[x, 5] + 1))
    elif data[x, 4] != 0:
        fa_rate[x] = (data[x, 4] / (data[x, 4] + data[x, 5]))

# calculate percent correct
per_cor = (data[:, 2] + data[:, 5]) / data[:, 1]

# calculate d_prime
d_prime = norm.ppf(hit_rate) - norm.ppf(fa_rate)

# Plot all hit rate, false alarm rate, and d prime
plt.figure(figsize=(10, 6))
plt.subplot(2, 2, 1)
plt.plot(sessions, hit_rate)
plt.ylabel('Hit Rate')
plt.subplot(2, 2, 2)
plt.plot(sessions, fa_rate)
plt.ylabel('FA rate')
plt.ylim([0, 1])
plt.subplot(2, 2, 3)
plt.plot(sessions, per_cor)
plt.ylabel('% Correct')
plt.ylim([0, 1])
plt.subplot(2, 2, 4)
plt.plot(sessions, d_prime)
plt.ylabel("D'")
plt.xlabel('Sessions')
plt.suptitle(ID + ' Performance', fontsize=16)
plt.show()






