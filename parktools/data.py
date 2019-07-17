'''
    Tools for querying and processing statcast data
'''
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pybaseball import statcast, statcast_batter, playerid_lookup


def filter_batted_balls(all_outcomes):
    """
    Filter out batted ball data from all_outcomes dataframe returned by get_batting_league() or get_batting_player()

    Arguments
        dataframe
    Returns
        dataframe
    """
    # Get BATTED BALLS ONLY from statcast output
    # when launch angle, launch speed, and hc_(x,y) are null, it's a strikeout, walk, etc
    batted_balls = all_outcomes.dropna()
    batted_balls.reset_index(inplace=True, drop=True)
    # add column to specify whether batted ball was a hit
    batted_balls = batted_balls.copy()
    batted_balls['hit'] = batted_balls['events'].isin(['single', 'double', 'triple', 'home_run'])
    
    # convert batter id to int
    batted_balls['batter'] = batted_balls['batter'].apply(int)
    return(batted_balls)
    

def get_batting_league(
    start_dt=None, end_dt=None, team=None, verbose=True, fname_all = None, fname_bb = None, features = [ 'events', 'description', 'batter', 'stand', 'launch_angle', 'launch_speed', 'hc_x', 'hc_y', 'pitcher', 'p_throws', 'pitch_type', 'release_speed', 'release_spin_rate']
):

    """
    Pull league-wide statcast batting data from baseballsavant using the pybaseball package
    https://github.com/jldbc/pybaseball
    https://baseballsavant.mlb.com/statcast_search

    Arguments
        start_dt: get data from start_dt forward
        stop_dt:  get data up to stop_dt
        team: search team only ('SEA', 'TEX', etc)
        verbose: display query status messages
        fname_all: export csv of all statcast at bat outcomes to this file
            **must be .csv**
        fname_bb: export csv of all outcomes with a batted ball to this file
            **must be .csv**

    Returns
        (all_outcomes, batted_balls) tuple of dataframes
            Saves to files 'fname_all' and 'fname_bb' if fname is not None
    """

    # get statcast data (this can take awhile)
    df = statcast(start_dt, end_dt, team, verbose)
    # discard null events
    all_outcomes = df[df['events'].notnull()]
    
    # get the specified features only
    all_outcomes = all_outcomes[features]
    
    if fname_all is not None:
        # export to csv
        all_outcomes.to_csv(fname_all, index=False)
        print('Exported: {}'.format(fname_all))
    
    # get batted balls only
    batted_balls = filter_batted_balls(all_outcomes)
    
    if fname_bb is not None:
        # export data
        batted_balls.to_csv(fname_bb, index=False)
        print('Exported: {}'.format(fname_bb))

    return(all_outcomes, batted_balls)


def get_batting_player(
    start_dt=None, end_dt=None, player_last='Vogelbach', player_first='Daniel', fname_all = None, fname_bb = None, features = ['events', 'description', 'batter', 'stand', 'launch_angle', 'launch_speed', 'hc_x', 'hc_y', 'pitcher', 'p_throws', 'pitch_type', 'release_speed', 'release_spin_rate']
): 

    """
    Pull player statcast batting data from baseballsavant using the pybaseball package
    https://github.com/jldbc/pybaseball
    https://baseballsavant.mlb.com/statcast_search

    Arguments
        start_dt: get data from start_dt forward
        stop_dt:  get data up to stop_dt
        player_id: mlb_am id
        fname_all: export csv of all statcast at bat outcomes to this file
            **must be .csv**
        fname_bb: export csv of all outcomes with a batted ball to this file
            **must be .csv**

    Returns
        (all_outcomes, batted_balls) tuple of dataframes
            Saves to files 'fname_all' and 'fname_bb' if fname is not None
    """

    # get player's mlbam_id (mlb advanced metrics id)
    # note: for players the same first+last name, this will get the
    # player who entered the league first
    # need to fix -- for now pick players with unique names
    # sorry Chris Davis :p
    player_id = playerid_lookup(
        player_last, player_first
    )['key_mlbam'].values[0]

    # get statcast data (this can take awhile)
    print('Querying batting stats for {} {}'.format(player_first,player_last))
    df = statcast_batter(start_dt, end_dt, player_id)
    # discard null events
    all_outcomes = df[df['events'].notnull()]
    
    # get the specified features only
    all_outcomes = all_outcomes[features]
    
    if fname_all is not None:
        # export to csv
        all_outcomes.to_csv(fname_all, index=False)
        print('Exported: {}'.format(fname_all))
    
    # get batted balls only
    batted_balls = filter_batted_balls(all_outcomes)
    
    if fname_bb is not None:
        # export data
        batted_balls.to_csv(fname_bb, index=False)
        print('Exported: {}'.format(fname_bb))

    return(all_outcomes, batted_balls)

def filter_batting_player(league_data='all_outcomes_2018.csv', player_last='Vogelbach', player_first='Daniel', fname=None):

    """
    This function should be used *after* getting league-wide data
    with get_league_batting_data(). If you already have league-wide data, this is faster than calling get_batting_player() since it doesn't query statcast data again 

    Arguments
        league_data: any .csv file returned from get_league_batting_data
        player_last: last name of player
        player_first: first name of player
        fname: name of .csv file to export
            defaults to [player_last]_[player_first].csv
    Returns
        dataframe
            Saves to file 'fname' if fname is not None
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
