import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn import neighbors
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap
from confusion import plot_confusion_matrix

def spray_angle(df):
    '''
        Calculate spray angle from hc_x and hc_y in statcast csv output
    '''
    # make home plate (0,0)
    hc_x = df['hc_x'] - 125.42
    hc_y = 198.27 - df['hc_y']
    df['spray_angle'] = np.arctan(hc_x/hc_y)
    df.loc[df['stand'] == 'L', 'spray_angle'] = -df.loc[df['stand'] == 'L', 'spray_angle']
    # convert to degrees
    df['spray_angle'] = df['spray_angle'].apply(np.rad2deg)
    return(df)

def pre_process(df):
    '''
          Process dataframe for KNN, break df into features and results
    '''
    # define features
    features = [
        'launch_speed', 'launch_angle', 'spray_angle'
    ]
    # add outcome column
    df['hit'] = (df['events'].isin(['single', 'double', 'triple', 'home_run']))
    # process df
    df = spray_angle(df)
    # break into features (X) and outcomes (y)
    X, y = df[features], df['hit']
    return(X, y)

def colorize(val):
    if val == True:
        rgb = tuple([0,0,1])
    else:
        rgb = tuple([1,0,0])
    return rgb



data = pd.read_csv('data/batted_balls_2018.csv')

X, y = pre_process(data)

X.describe()

# split up data into train, test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state = 0
)

# K-Nearest Neighbors 
nn = 1
clf = neighbors.KNeighborsClassifier(nn)
clf.fit(X, y)

print('Accuracy of KNN classifier on training set: {:.2f}'
     .format(clf.score(X_train, y_train)))
print('Accuracy of KNN classifier on test set: {:.2f}'
     .format(clf.score(X_test, y_test)))

# calculate expected batting average for player
dee = pd.read_csv('data/dee_gordon_2019.csv')

X0, y0 = pre_process(dee)

predicted_outcomes = clf.predict(X0)
unique, counts = np.unique(predicted_outcomes, return_counts=True)
d = dict(zip(unique, counts))
xBABIP = d[True]/(d[False] + d[True])

hits = 0
for event in ['single', 'double', 'triple']:
    hits += dee['events'].value_counts()[event]

BABIP = hits/len(dee['events'])

print('\nPredicted BABIP: {:.3f}'.format(xBABIP))
print('Actual BABIP: {:.3f}'.format(BABIP))

plot_confusion_matrix(y0, predicted_outcomes, [0, 1], normalize=True)
plt.show()
# Plot results
# Create color maps
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

colors = y.apply(colorize)

fig = plt.figure(figsize=(9,9))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(
    X['launch_speed'], X['launch_angle'], X['spray_angle'],
    c=colors, s=0.001
)
ax.set_xlabel('Launch Speed (MPH)')
ax.set_ylabel('Launch Angle (deg)')
ax.set_zlabel('Spray Angle (deg)')
plt.show()
