#!/usr/bin/env python3
# coding: utf-8
'''
Construct the KNN model. Train and test with 2018 batting data, then predict
2019 batting averages for the two players selected in
get_player_batting_data.py. Plot confusion matrices for both players. Save the
model for later use.

Note: see the `data/nn_tests/` folder for some rational for selecting N=31
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
fig, ax = plot_confusion_matrix(
    y_edwin, predicted_outcomes_edwin, ['out', 'hit'], normalize=True,
    title='Edwin Encarnacion, N = {}'.format(nn)
)
fig.savefig('plots/confusion_matrix_edwin_N{}.png'.format(nn))
# plt.show()

# save the model
pickle.dump(clf, open('xBA_knn_model.pickle', 'wb'))
