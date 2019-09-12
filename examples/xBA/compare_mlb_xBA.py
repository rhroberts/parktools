#!/usr/bin/env python3
# coding: utf-8
'''
Compare our results from calc_player_xBAs_2019.py with the official
MLB/baseballsavant xBA values.
'''
import pandas as pd

calc_xBA = pd.read_csv('data/xBA_results_2019.csv', index_col=0)
statcast_xBA = pd.read_csv('ref_data/expected_stats_baseballsavant.csv')

# clean up and join dataframes
calc_xBA.drop('BA', axis=1, inplace=True)
calc_xBA.columns = ['first_name', 'last_name', 'xBA_calc']
statcast_xBA.set_index('player_id', drop=True, inplace=True)
statcast_xBA = statcast_xBA[[' first_name', 'last_name', 'est_ba', 'ba']]
statcast_xBA.columns = ['first_name', 'last_name', 'xBA_mlb', 'actual_BA']
xBA_compare = calc_xBA.join(
    statcast_xBA, on=calc_xBA.index, how='left', rsuffix='_'
)
# drop duplicate columns
xBA_compare.drop(['first_name_', 'last_name_'], axis=1, inplace=True)
xBA_compare.sort_values('actual_BA', ascending=False, inplace=True)
xBA_compare = xBA_compare.round(3)
print(xBA_compare)
# calculate average difference between calculated and MLB xBA
diff = (xBA_compare['xBA_calc'] -
        xBA_compare['xBA_mlb']).abs().sum()/xBA_compare.shape[0]
diff = round(diff, 3)
print('\nAverage difference between calculated and MLB xBA: {}'.format(diff))
# export results
fname = 'data/xBA_compare_2019.csv'
xBA_compare.to_csv(fname)
print("xBA results exported to '{}'".format(fname))
