#!/usr/bin/env python3
# coding: utf-8
'''
# Expected Batting Average

## About

- This project aims to replicate the MLB statcast advanced stat "xBA"

## Goals

1) Use supervised ML to predict whether a batted ball is a hit ot an out based
on:
    - Exit velo
    - Launch angle
    - Spray angle
    - Note: other features from statcast data ended up not improving the model,
        though MLB did recently started using some sort of sprint speed in
        their xBA calculations

2) Find the *expected* batting average for a batter given the above parameters
for each batted ball in play

3) Compare results with statcast's xBA results

## Data

- Data gathered from baseball savant (statcast) search
- Example search query to get all batted balls resulting in outs in 2018
    - https://baseballsavant.mlb.com/statcast_search?hfPT=&hfAB=single
    %7Cdouble%7Ctriple%7Chome%5C.%5C.run%7Cfield%5C.%5C.out%7Cstrikeout
    %7Cstrikeout%5C.%5C.double%5C.%5C.play%7Cdouble%5C.%5C.play%7Cgrounded
    %5C.%5C.into%5C.%5C.double%5C.%5C.play%7Cfielders%5C.%5C.choice%7Cfielders
    %5C.%5C.choice%5C.%5C.out%7Cforce%5C.%5C.out%7Csac%5C.%5C.bunt%7Csac
    %5C.%5C.bunt%5C.%5C.double%5C.%5C.play%7Csac%5C.%5C.fly%7Csac%5C.%5C.fly
    %5C.%5C.double%5C.%5C.play%7Ctriple%5C.%5C.play%7C&hfBBT=&hfPR=&hfZ=
    &stadium=&hfBBL=&hfNewZones=&hfGT=R%7C&hfC=&hfSea=2018%7C&hfSit=
    &player_type=batter&hfOuts=&opponent=&pitcher_throws=&batter_stands=
    &hfSA=&game_date_gt=&game_date_lt=&hfInfield=&team=&position=&hfOutfield=
    &hfRO=&home_road=&hfFlag=&hfPull=&metric_1=&hfInn=&min_pitches=0
    &min_results=0&group_by=name&sort_col=pitches
    &player_event_sort=h_launch_speed&sort_order=desc&min_pas=0#results
    - Note: seems like this returns a maximum of 40,000 results
- Data reference
    - https://baseballsavant.mlb.com/csv-docs
- See get_league_batting_data.py

## Notes

- My plan is to use 2018 results in the training/test sets to determine 2019
xBA results
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
plt.show()
fig.savefig('plots/feature_exploration.png', dpi=300)
