'''
    Tools for querying and processing statcast data
'''
import pandas as pd
import numpy as np
from pybaseball import statcast, statcast_batter, playerid_lookup


def filter_batted_balls(all_outcomes):
    """
    Filter batted ball data from all_outcomes dataframe returned by
    get_batting_league() or get_batting_player()

    Arguments
        dataframe
    Returns
        dataframe
    """
    # Get BATTED BALLS ONLY from statcast output
    # if launch angle, launch speed, and hc_(x,y) are null, it's a K, walk, etc
    batted_balls = all_outcomes.dropna()
    batted_balls.reset_index(inplace=True, drop=True)
    # add column to specify whether batted ball was a hit
    batted_balls = batted_balls.copy()
    batted_balls['hit'] = batted_balls['events'].isin([
        'single', 'double', 'triple', 'home_run'
    ])

    # convert batter id to int
    batted_balls['batter'] = batted_balls['batter'].apply(int)
    return(batted_balls)


def get_batting_league(
    start_dt=None, end_dt=None, team=None, verbose=True, fname_all=None,
    fname_bb=None, features=[
        'events', 'description', 'batter', 'stand', 'launch_angle',
        'launch_speed', 'hc_x', 'hc_y', 'pitcher', 'p_throws', 'pitch_type',
        'release_speed', 'release_spin_rate'
    ]
):

    """
    Pull league-wide statcast batting data from baseballsavant using pybaseball
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
            Saves to files 'fname_all' and 'fname_bb' if fname is not None.
            Note that all_outcomes DOES NOT include plate appearances that do
            not count as at bats.
            TODO: output truly all outcomes, all at bats, and all batted balls
            as three separate files.
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
    start_dt=None, end_dt=None, player_last='Vogelbach', player_first='Daniel',
    fname_all=None, fname_bb=None, features=[
        'events', 'description', 'batter', 'stand', 'launch_angle',
        'launch_speed', 'hc_x', 'hc_y', 'pitcher', 'p_throws', 'pitch_type',
        'release_speed', 'release_spin_rate'
    ]
):

    """
    Pull player statcast batting data from baseballsavant using pybaseball
    https://github.com/jldbc/pybaseball
    https://baseballsavant.mlb.com/statcast_search

    Arguments
        start_dt: get data from start_dt forward
        stop_dt:  get data up to stop_dt
        player_last: player's last name
        player_first: player's first name
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
    print('Querying batting stats for {} {}'.format(player_first, player_last))
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


def filter_batting_player(league_data='all_outcomes_2018.csv',
                          player_last='Vogelbach', player_first='Daniel',
                          fname=None):

    """
    This function should be used *after* getting league-wide data
    with get_league_batting_data(). If you already have league-wide data, this
    is faster than calling get_batting_player() since it doesn't query statcast
    data again

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
    # need to fix -- for now pick players with unique names, sorry Chris Davis
    mlbam_id = playerid_lookup(
        player_last, player_first
    )['key_mlbam'].values[0]

    league_df = pd.read_csv(league_data)
    player_df = league_df[league_df['batter'] == mlbam_id]
    player_df.reset_index(inplace=True, drop=True)
    if fname is not None:
        player_df.to_csv(fname, index=False)
    return(player_df)


def spray_angle(outcomes):
    '''
        Calculate spray angle from hc_x and hc_y columns in outcomes dataframe
        Adds returns the same outcomes dataframe within an added 'spray_angle'
        column

        ref: https://tht.fangraphs.com/research-notebook-new-format-for-
                                    statcast-data-export-at-baseball-savant/
    '''
    # make home plate (0,0)
    hc_x = outcomes['hc_x'] - 125.42
    hc_y = 198.27 - outcomes['hc_y']
    outcomes['spray_angle'] = np.arctan(hc_x/hc_y)
    outcomes.loc[outcomes['stand'] == 'L', 'spray_angle'] = \
        -outcomes.loc[outcomes['stand'] == 'L', 'spray_angle']
    # convert to degrees
    outcomes['spray_angle'] = outcomes['spray_angle'].apply(np.rad2deg)
    return(outcomes)


def pre_process(outcomes, features=[
    'launch_speed', 'launch_angle', 'spray_angle'
], label='hit'):
    '''
    Process outcomes dataframe for a machine learning model

    Arguments
        outcomes: dataframe - returned by get_batting_league(),
                              get_batting_player(), etc
        features: array - columns from outcomes to use as the label (in the
                       machine learning sense) in the model
        label: str - column to use as outcome
    '''
    # define hits and outs that count as an at-bat
    hits = ['single', 'double', 'triple', 'home_run']
    outs_AB = [
        'strikeout', 'field_out', 'grounded_into_double_play',
        'force_out', 'double_play', 'field_error', 'fielders_choice',
        'fielders_choice_out', 'batter_interference',
        'strikeout_double_play', 'triple_play'
    ]
    # copy to new dataframe
    outcomes = outcomes.copy()
    # !! this is imposing the ba/babip stuff
    outcomes = outcomes[outcomes.events.isin([*hits, *outs_AB])]
    # add hit columns if default label='hit' is used
    if label == 'hit':
        # add hit outcome column (0: out_AB, 1: hit)
        outcomes[label] = 0
        outcomes.loc[outcomes.events.isin(hits), label] = 1
    # calculate spray angle from hc_x and hc_y if it not already in outcomes
    if 'spray_angle' not in outcomes:
        outcomes = spray_angle(outcomes)
    # set launch/spray angle/speed = 0 for strikeouts
    # these events count for *most* of the NaN values here
    # check with:
    #   outcomes.loc[outcomes.launch_speed.isnull(), 'events'].value_counts()
    outcomes.loc[
        outcomes.events.isin(['strikeout', 'strikeout_double_play']),
        [*features, 'hc_x', 'hc_y']
    ] = 0
    # drop remaining rows with NaN
    outcomes.dropna(inplace=True)
    # break into features (X) and outcomes (y)
    X, y = outcomes[features], outcomes[[label]]
    return(X, y, outcomes)
