#!/usr/bin/env python3
'''
Get batting data for entire league, save files with all outcomes and with
batted balls only.
'''
from parktools.data import get_batting_league

# get statcast batting data for 2018 season
all_outcomes_2018, batted_balls_2018 = get_batting_league(
    '2018-03-29', '2018-10-01'
)

# get statcast batting data for 2019 season to date
all_outcomes_2019, batted_balls_2019 = get_batting_league(
    '2019-03-20', '2019-09-10'
)

# export dataframes to csv
all_outcomes_2018.to_csv('data/all_outcomes_2018.csv')
all_outcomes_2019.to_csv('data/all_outcomes_2019.csv')
batted_balls_2018.to_csv('data/batted_balls_2018.csv')
batted_balls_2019.to_csv('data/batted_balls_2019.csv')

# preview results
print('All Batting Outcomes (2018)')
print(all_outcomes_2018.head())
print('All Batting Outcomes (2019)')
print(all_outcomes_2019.head())
print('Batted Balls Only (2018)')
print(batted_balls_2018.head())
print('Batted Balls Only (2019)')
print(batted_balls_2019.head())
