import sys
import numpy as np
import pandas as pd
import os
import scipy as sp
from scipy.stats import norm
import matplotlib.pyplot as plt


def count_licks(file_name):
    numpy_data = np.loadtxt(file_name, dtype="str", delimiter="\0")
    # Step 1. Find all the instances where a trial starts
    indices = []
    tt = []
    for idx, string in enumerate(numpy_data):
        if string.find('TRIAL') != -1:
            indices.append(idx)
        if string.find('TT49') != -1:
            tt.append(1)
        elif string.find('TT48') != -1:
            tt.append(0)

    # Step 2. Iterate through appropriate lines and count the number of licks
    licks = []
    for idx, i in enumerate(indices):
        if idx == len(indices) - 1:
            count = str(numpy_data[i:-1]).count('LICK')
            licks.append(count)
        else:
            count = str(numpy_data[i:indices[idx + 1]]).count('LICK')
            licks.append(count)

    data = np.column_stack([np.array(tt), np.array(licks)])
    return data


ID = input('Enter mouse ID: ')
path = input('Enter folder path: ')
os.chdir(path)
all_licks = {}  # use a library to store the lick rates at each session
for file in os.listdir():
    if file.endswith(".txt"):
        file_path = f"{path}\\{file}"
        all_licks[file] = count_licks(file_path)

# Calculate lick rate for Go and No Go trials
go_lick_rate = []
no_go_lick_rate = []
for i in list(all_licks.keys()):
    matrix = all_licks[i]
    go_lick_rate.append(np.mean(matrix[matrix[:, 0] == 1, 1]))
    no_go_lick_rate.append(np.mean(matrix[matrix[:, 0] == 0, 1]))

print(go_lick_rate)
print(no_go_lick_rate)

# Plot stuff
sessions = range(1, len(all_licks)+1)
plt.plot(sessions, go_lick_rate, 'b', label='Go Trials')
plt.plot(sessions, no_go_lick_rate, 'g', label='No Go Trials')
plt.xlabel('Sessions')
plt.ylabel('Lick Rate (licks per trial)')
plt.legend()
plt.title(str(ID) + ' performance')
plt.show()

