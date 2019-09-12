#!/usr/bin/env python3
'''
Get batting data for specific players to use in model testing. We select two
very different hitter types (fast, contact hitter & slow, power hitter) to make
sure the model can adequately represent both extremes.
'''
from parktools.data import get_batting_player

dee_2019, _ = get_batting_player(
    '2019-03-20', '2019-09-10', 'Gordon', 'Dee',
    'data/dee_gordon_2019.csv'
)
edwin_2019, _ = get_batting_player(
    '2019-03-20', '2019-09-10', 'Encarnacion', 'Edwin',
    'data/edwin_encarnacion_2019.csv'
)

print(dee_2019)
print(edwin_2019)
