'''
    Tools for calculating xBA
'''
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pybaseball import statcast, playerid_lookup

def get_league_batting_data(start_dt='2018-03-29',
                            end_dt='2018-10-01',
                            # fname_all='all_outcomes_2018.csv',
                            # fname_bb='batted_balls_2018.csv',
                            fname_all = None,
                            fname_bb = None,
                            features = [
                                'events',
                                'description',
                                'batter',
                                'stand',
                                'launch_angle',
                                'launch_speed',
                                'hc_x',
                                'hc_y',
                                'pitcher',
                                'p_throws',
                                'pitch_type',
                                'release_speed',
                                'release_spin_rate']):

    """
    Pull league-wide statcast batting data from baseballsavant using the pybaseball package
    https://github.com/jldbc/pybaseball
    https://baseballsavant.mlb.com/statcast_search

    Arguments
        start_dt: get data from start_dt forward
        stop_dt:  get data up to stop_dt
        fname_all: export csv of all statcast at bat outcomes to this file
            **must be .csv**
        fname_bb: export csv of all outcomes with a batted ball to this file
            **must be .csv**

    Returns
        (all_outcomes, batted_balls) tuple of dataframes
    """

    # get statcast data (this can take awhile)
    df = statcast(start_dt, end_dt)
    # discard null events
    all_outcomes = df[df['events'].notnull()]
    
    # get the specified features only
    all_outcomes = all_outcomes[features]
    
    if fname_all is not None:
        # export to csv
        all_outcomes.to_csv(fname_all, index=False)
        print('Exported: {}'.format(fname_all))
    
    # BATTED BALLS ONLY
    # when launch angle, launch speed, and hc_(x,y) are null, it's a strikeout, walk, etc
    batted_balls = all_outcomes.dropna()
    batted_balls.reset_index(inplace=True, drop=True)
    # add column to specify whether batted ball was a hit
    batted_balls = batted_balls.copy()
    batted_balls['hit'] = batted_balls['events'].isin(['single', 'double', 'triple', 'home_run'])
    
    # convert batter id to int
    batted_balls['batter'] = batted_balls['batter'].apply(int)
    
    if fname_bb is not None:
        # export data
        batted_balls.to_csv(fname_bb, index=False)
        print('Exported: {}'.format(fname_bb))

    return(all_outcomes, batted_balls)


def filter_player_batting_data(league_data='all_outcomes_2018.csv',
                            player_last='gordon',
                            player_first='dee',
                            fname=None):

    """
    This function should be used *after* getting league-wide data
    with get_league_batting_data(). With pybaseball.statcast you can't
    query by player, only league-wide or team. To avoid another long query,
    it's best to get the full league dataset first, then use this function
    to filter for a specific player

    Arguments
        league_data: any .csv file returned from get_league_batting_data
        player_last: last name of player
        player_first: first name of player
        fname: name of .csv file to export
            defaults to [player_last]_[player_first].csv
    Returns
        csv file 'fname'
    """

    # get player's mlbam_id (mlb advanced metrics id)
    # note: for players the same first+last name, this will get the
    # player who entered the league first
    # need to fix -- for now pick players with unique names
    # sorry Chris Davis :p
    mlbam_id = playerid_lookup(
        player_last, player_first
    )['key_mlbam'].values[0]

    league_df = pd.read_csv(league_data)
    player_df = league_df[league_df['batter'] == mlbam_id]
    player_df.reset_index(inplace=True, drop=True)
    if fname is not None:
        player_df.to_csv(fname, index=False)
    return(player_df)


def calc_BABIP(batted_ball_data):
    """
    Calculate batting average on balls in play from statcast batted ball data returned from get_league_batting_data() or get_player_batting_data()
    This doesn't always perfectly match published results, but it's a good sanity check to make sure results are close to actual values

    Arguments
        batted_balls_data: .csv output from get_league_batting_data() or filter_player_batting_data()

    Returns
        float BABIP
    """

    df = pd.read_csv(batted_ball_data)
    # BABIP = (Hits - HR)/(AB-K-HR-SF)
    counts = df['events'].value_counts()
    BABIP = (
        (counts['single'] + counts['double'] + counts['triple'])/\
                (counts.sum() - counts['home_run'])
    )
    return(BABIP)


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Purples):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Adapted from: https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py

    Arguments
        y_true: test set outcomes
        y_pred: training set predictions
        title: plot title
        cmap: matplotlib colormap to use for plot

    Returns
        mpl axis (ax)
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    # classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax
