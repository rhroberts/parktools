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
- See:
    - get_league_batting_data.py
    - feature_exploration.py

## Notes

- My plan is to use 2018 results in the training/test sets to determine 2019
xBA results
'''

import pickle
import pandas as pd
import numpy as np
from parktools.data import pre_process
from parktools.tools import calc_BA, plot_confusion_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import neighbors

# import and process batting data
data = pd.read_csv('data/all_outcomes_2018.csv', index_col=0)
X, y, df = pre_process(data)

# split up data into train, test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=0
)

# K-Nearest Neighbors
nn = 31
clf = neighbors.KNeighborsClassifier(nn)
clf.fit(X, y)

print('Accuracy of KNN classifier on training set: {:.2f}'
      .format(clf.score(X_train, y_train)))
print('Accuracy of KNN classifier on test set: {:.2f}'
      .format(clf.score(X_test, y_test)))

# calculate expected batting average for a couple players
dee = pd.read_csv('data/dee_gordon_2019.csv')
edwin = pd.read_csv('data/edwin_encarnacion_2019.csv')

X_dee, y_dee, df_dee = pre_process(dee)
X_edwin, y_edwin, df_edwin = pre_process(edwin)

predicted_outcomes_dee = clf.predict(X_dee)
unique, counts = np.unique(predicted_outcomes_dee, return_counts=True)
d = dict(zip(unique, counts))
# hit=1, out=0
xBA_dee = d[1]/(d[0] + d[1])

predicted_outcomes_edwin = clf.predict(X_edwin)
unique, counts = np.unique(predicted_outcomes_edwin, return_counts=True)
d = dict(zip(unique, counts))
xBA_edwin = d[1]/(d[0] + d[1])

BA_dee = calc_BA('data/dee_gordon_2019.csv')
BA_edwin = calc_BA('data/edwin_encarnacion_2019.csv')

print('\nExpected Batting Average for Dee Gordon: {:.3f}'.format(xBA_dee))
print('Actual Batting Average: {:.3f}\n'.format(BA_dee))
print('\nExpected Batting Average for Edwin Encarnacion: {:.3f}'
      .format(xBA_edwin))
print('Actual Batting Average: {:.3f}\n'.format(BA_edwin))

plt.style.use('fivethirtyeight')
fig, ax = plot_confusion_matrix(
    y_dee, predicted_outcomes_dee, ['out', 'hit'], normalize=True,
    title='Dee Gordon, N = {}'.format(nn)
)
fig.savefig('plots/confusion_matrix_dee_N{}.png'.format(nn))
plot_confusion_matrix(
    y_edwin, predicted_outcomes_edwin, ['out', 'hit'], normalize=True,
    title='Edwin Encarnacion, N = {}'.format(nn)
)
fig.savefig('plots/confusion_matrix_edwin_N{}.png'.format(nn))
plt.show()

# save the model
pickle.dump(clf, open('xBA_knn_model.pickle', 'wb'))
