#!/usr/bin/env python
# coding: utf-8

'''
Pull league-wide 2018 and 2019 statcast batting data from baseballsavant using the pybaseball package
https://github.com/jldbc/pybaseball
https://baseballsavant.mlb.com/statcast_search

Data from this script is exported to:
    data/all_outcomes_2018.csv
    data/all_outcomes_2019.csv
    data/batted_balls_2018.csv
    data/batted_balls_2019.csv
'''

import pandas as pd
from pybaseball import statcast

df_2018 = statcast(start_dt='2018-03-29', end_dt='2018-10-01')
df_2019 = statcast(start_dt='2019-03-20', end_dt='2019-07-10')
outcomes_2018 = df_2018[df_2018['events'].notnull()]
outcomes_2019 = df_2019[df_2019['events'].notnull()]

# choose the features to include 
features = [
    'events', 'description', 'batter', 'stand', 'launch_angle', 'launch_speed', 'hc_x', 'hc_y', 'pitcher', 'p_throws', 'pitch_type', 'release_speed', 'release_spin_rate' 
]

select_2018 = outcomes_2018[features]
select_2019 = outcomes_2019[features]

select_2018.to_csv('data/all_outcomes_2018.csv', index=False)
select_2019.to_csv('data/all_outcomes_2019.csv', index=False)

# when launch angle, launch speed, and hc_(x,y) are null, it's a strikeout, walk, etc
batted_balls_2018 = select_2018.dropna()
batted_balls_2018.reset_index(inplace=True, drop=True)
# drop sacrifices that don't affect BA
# batted_balls_2018 = batted_balls_2018[
#     ~batted_balls_2018.events.isin(['sac_fly', 'sac_bunt', 'sac_fly_double_play', 'sac_bunt_double_play'])
# ]
batted_balls_2018 = batted_balls_2018.copy()
batted_balls_2018['hit'] = batted_balls_2018['events'].isin(['single', 'double', 'triple', 'home_run'])

# BABIP = (Hits - HR)/(AB-K-HR-SF)
c = batted_balls_2018['events'].value_counts()
BABIP = (
    (c['single'] + c['double'] + c['triple'])/(c.sum() - c['home_run'])
)

# this is consistent with league-wide BABIP in 2018, which was 0.296 according to fangraphs
# good sanity check, removing null values didn't skew things too much
print("\n2018 BABIP: {:f}".format(BABIP))

# convert batter id to int
batted_balls_2018['batter'] = batted_balls_2018['batter'].apply(int)

# export data
batted_balls_2018.to_csv('data/batted_balls_2018.csv', index=False)

# REPEAT for 2019

# when launch angle, launch speed, hc_(x,y), are null, it's a strikeout, walk, etc
batted_balls_2019 = select_2019.dropna()
batted_balls_2019.reset_index(inplace=True, drop=True)
batted_balls_2019 = batted_balls_2019.copy()
batted_balls_2019['hit'] = batted_balls_2019['events'].isin(['single', 'double', 'triple', 'home_run'])

# BABIP = (Hits - HR)/(AB-K-HR-SF)
c = batted_balls_2019['events'].value_counts()
BABIP = (
    (c['single'] + c['double'] + c['triple'])/(c.sum() - c['home_run'])
)

# Consistent with 2019 league-wide BABIP (.293)
print("\n2019 BABIP: {:f}".format(BABIP))

# convert batter id to int
batted_balls_2019['batter'] = batted_balls_2019['batter'].apply(int)

# export data
batted_balls_2019.to_csv('data/batted_balls_2019.csv', index=False)

print('Finished exporting:')
print('Exported:')
print('\tdata/all_outcomes_2018.csv')
print('\tdata/all_outcomes_2019.csv')
print('\tdata/batted_balls_2018.csv')
print('\tdata/batted_balls_2019.csv')
