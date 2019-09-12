#!/usr/bin/env python3
# coding: utf-8
'''
Visualize relationships between the key features of the dataset:
    1. Exit velocity
    2. Launch angle
    3. Spray angle
'''
import pandas as pd
from parktools.data import pre_process
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

# import and process batting data
data = pd.read_csv('data/all_outcomes_2018.csv', index_col=0)
X, y, df = pre_process(data)

plt.style.use('fivethirtyeight')
fig, ax = plt.subplots(2, 2, figsize=(12, 12))

# plot outs as red, hits as blue
colors = pd.DataFrame(y.copy())
colors['colors'] = 'red'
colors.loc[colors['hit'] == 1, 'colors'] = 'blue'
colors = np.array(colors['colors'])

# launch angle vs. launch speed
ax[0, 0].scatter(
    X['launch_angle'], X['launch_speed'], s=20, c=colors,
    edgecolors='black', linewidth=1
)
ax[0, 0].set_xlabel('Launch Angle (deg)')
ax[0, 0].set_ylabel('Launch Speed (MPH)')

# spray angle vs. launch speed
ax[0, 1].scatter(
    X['spray_angle'], X['launch_speed'], s=20, c=colors,
    edgecolors='black', linewidth=1
)
ax[0, 1].set_xlabel('Spray Angle (deg)')
ax[0, 1].set_ylabel('Launch Speed (MPH)')

# spray angle vs. launch speed
ax[1, 0].scatter(
    X['launch_angle'], X['spray_angle'],  s=20, c=colors,
    edgecolors='black', linewidth=1
)
ax[1, 0].set_xlabel('Launch Angle (deg)')
ax[1, 0].set_ylabel('Spray Angle (deg)')

# 3D plot of all features
for val in ['top', 'bottom', 'left', 'right']:
    ax[1, 1].spines[val].set_visible(False)
ax[1, 1].tick_params(
    axis='x', which='both',
    top=False, bottom=False
)
ax[1, 1].tick_params(
    axis='y', which='both',
    left=False, right=False
)
ax[1, 1].set_yticklabels('')
ax[1, 1].set_xticklabels('')
ax = fig.add_subplot(224, projection='3d')

ax.scatter(
    X['launch_angle'], X['launch_speed'], X['spray_angle'],
    c=colors, s=1,
)
ax.set_ylim([10, 100])
ax.set_xlim([-20, 100])
ax.set_zlim([-20, 60])

ax.set_ylabel('Launch Speed (MPH)')
ax.set_xlabel('Launch Angle (deg)')
ax.set_zlabel('Spray Angle (deg)')
fig.tight_layout()
# plt.show()
fig.savefig('plots/feature_exploration.png', dpi=300)
