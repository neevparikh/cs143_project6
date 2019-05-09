#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

data = np.load("pose_data/normed_source_pose.npy", allow_pickle=True)
subsets = np.load("pose_data/test_subsets.npy", allow_pickle=True)

indexes = []
complete = []



for i, d in enumerate(data.tolist()):
    d = np.array(d)
    if d.shape[0] == (18, 4):
        indexes.append(i)
        complete.append(d)

complete = np.array(complete)
print(complete[8])
plt.plot(complete[:,0,0])
filtered = savgol_filter(complete, window_length=30, polyorder=3, axis=0)
plt.plot(filtered[:0,0])
plt.show()
