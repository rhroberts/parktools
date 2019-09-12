# Expected Batting Average

## About

- This project aims to replicate the MLB statcast advanced stat "xBA"

## Goals

1) Use supervised machine learning to predict whether a batted ball is a hit or an out based on
    - Exit velo
    - Launch angle
    - Spray angle
    - Note: I found that other features from statcast data ended up not improving the model, though MLB did recently started using some sort of sprint speed in their xBA calculations

2) Find the *expected* batting average for a batter given the above parameters for each batted ball in play

3) Compare results with statcast's xBA results

## Data

- Data gathered from baseball savant (statcast) search
- Example search query to get all batted balls resulting in outs in 2018
    - https://baseballsavant.mlb.com/statcast_search?hfPT=&hfAB=single%7Cdouble%7Ctriple%7Chome%5C.%5C.run%7Cfield%5C.%5C.out%7Cstrikeout%7Cstrikeout%5C.%5C.double%5C.%5C.play%7Cdouble%5C.%5C.play%7Cgrounded%5C.%5C.into%5C.%5C.double%5C.%5C.play%7Cfielders%5C.%5C.choice%7Cfielders%5C.%5C.choice%5C.%5C.out%7Cforce%5C.%5C.out%7Csac%5C.%5C.bunt%7Csac%5C.%5C.bunt%5C.%5C.double%5C.%5C.play%7Csac%5C.%5C.fly%7Csac%5C.%5C.fly%5C.%5C.double%5C.%5C.play%7Ctriple%5C.%5C.play%7C&hfBBT=&hfPR=&hfZ=&stadium=&hfBBL=&hfNewZones=&hfGT=R%7C&hfC=&hfSea=2018%7C&hfSit=&player_type=batter&hfOuts=&opponent=&pitcher_throws=&batter_stands=&hfSA=&game_date_gt=&game_date_lt=&hfInfield=&team=&position=&hfOutfield=&hfRO=&home_road=&hfFlag=&hfPull=&metric_1=&hfInn=&min_pitches=0&min_results=0&group_by=name&sort_col=pitches&player_event_sort=h_launch_speed&sort_order=desc&min_pas=0#results
    - Note: This returns a maximum of 40,000 results, so it's best to query programmatically
        - via `parktools.data.get_league_batting_data()`
- Data reference
    - https://baseballsavant.mlb.com/csv-docs
- See:
    - get_league_batting_data.py
    - feature_exploration.py

## Notes

- I use 2018 results in the training/test sets to determine 2019 xBA results

## Results

- Execute `run_all` to get all of the demo results. It will take some time -- possibly more than an hour.
