#!/usr/bin/env python3
# coding: utf-8
'''
Using the model from model_text.py, calculate 2019 expected batting averages
for players with > 300 at bats.
'''
import pickle
import pandas as pd
import numpy as np
from pybaseball import playerid_reverse_lookup
from parktools.data import pre_process
from parktools.tools import calc_BA

# get the model from `model_test.py`
clf = pickle.load(open('xBA_knn_model.pickle', 'rb'))
# get 2019 batting data
data2019 = pd.read_csv('data/all_outcomes_2019.csv', index_col=0)
# remove players with fewer than 300 ABs
AB300 = []
for pid in data2019['batter'].unique():
    if data2019.loc[
        data2019['batter'] == pid
    ].shape[0] >= 300 and pid not in AB300:
        AB300.append(pid)

cBA, xBA = [], []
xBA_dict = {}
for i, pid in enumerate(AB300):
    print('Processing {:d}/{:d}'.format(i+1, len(AB300)) + ' Player IDs...')
    # get player name from player_id
    plast, pfirst = playerid_reverse_lookup(
        [pid], key_type='mlbam'
    ).iloc[0, :2]
    print('Player ID {} --> {} {}'.format(pid, pfirst, plast))
    # filter player data from league-wide batting data
    pdata = data2019.loc[data2019['batter'] == pid].copy()
    Xp, yp, dfp = pre_process(pdata)
    # calculate xBA from model
    predicted_outcomes = clf.predict(Xp)
    unique, counts = np.unique(predicted_outcomes, return_counts=True)
    d = dict(zip(unique, counts))
    # hit=1, out=0
    xBA = d[1]/(d[0] + d[1])
    # calculate standard BA
    cBA = calc_BA(dfp, from_file=False)
    xBA_dict[pid] = [pfirst, plast, round(cBA, 3), round(xBA, 3)]

# convert to datafrom and save results to csv
xBA_df = pd.DataFrame.from_dict(xBA_dict, orient='index')
xBA_df.columns = ['first_name', 'last_name', 'BA', 'xBA']
xBA_df.index = [int(x) for x in xBA_df.index]
xBA_df.index.name = 'player_id'
xBA_df.sort_values('BA', inplace=True, ascending=False)
print('\n{}'.format(xBA_df))
xBA_df.to_csv('data/xBA_results_2019.csv')
