import numpy as np
import os
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


def analyze_folder(path):
    folder_file_data = []                      # make list, each element is a matrix for a given file in the folder
    go_lick_rate = []
    no_go_lick_rate = []
    os.chdir(path)
    for file in os.listdir():
        if file.endswith(".txt"):
            file_path = f"{path}\\{file}"
            folder_file_data.append(count_licks(file_path))  # collect matrices for each session and
    for matrix in folder_file_data:
        go_lick_rate.append(np.mean(matrix[matrix[:, 0] == 1, 1]))
        no_go_lick_rate.append(np.mean(matrix[matrix[:, 0] == 0, 1]))
    return np.column_stack([go_lick_rate, no_go_lick_rate])


data_set = input('Enter dataset path: ')     # Enter folder with all data
subdir = list(os.walk(data_set))             # Create a list of all data folders
file_names = subdir[0][1]                    # Get a list of folder names
del subdir[0]                                # Remove top branch of tree

total_mouse_data = []
for folder, i in enumerate(subdir):
    total_mouse_data.append(analyze_folder(subdir[folder][0]))

sessions = len(total_mouse_data[0][:, 0])           # Arbitrarily get the number of sessions from the first mouse
go_data = np.zeros((sessions, len(file_names)))
no_go_data = np.zeros((sessions, len(file_names)))
for idx, mouse_data in enumerate(total_mouse_data):
    go_data[:, idx] = mouse_data[:, 0]
    no_go_data[:, idx] = mouse_data[:, 1]

mean_go = np.mean(go_data, axis=1)
mean_no_go = np.mean(no_go_data, axis=1)
sd_go = np.std(go_data, axis=1)
sd_no_go = np.std(no_go_data, axis=1)

plt.errorbar(range(sessions), mean_go, sd_go, label='Go Trials')
plt.errorbar(range(sessions), mean_no_go, sd_no_go, label='No Go Trials')
plt.xlabel('Session Number')
plt.ylabel('Lick Rate (Licks/trial)')
plt.title('Total Cohort Performance (n = 5)')
plt.legend()
plt.show()
