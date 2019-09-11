'''
    Tools for calculating xBA
'''
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def calc_BABIP(batted_ball_data):
    """
    Calculate batting average on balls in play from statcast batted ball data
    returned from get_league_batting_data() or get_player_batting_data().
    This doesn't always perfectly match published results, but it's a good
    sanity check to make sure results are close to actual values.

    Arguments
        batted_balls_data: .csv output from get_league_batting_data() or
                           filter_player_batting_data()

    Returns
        float BABIP
    """

    df = pd.read_csv(batted_ball_data)
    # BABIP = (Hits - HR)/(AB-K-HR-SF)
    counts = df['events'].value_counts()
    BABIP = (
        (counts['single'] + counts['double'] + counts['triple']) /
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
    Adapted from: "https://scikit-learn.org/stable/auto_examples/
        model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-
        model-selection-plot-confusion-matrix-py"

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
