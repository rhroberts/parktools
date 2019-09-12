import parktools.data as ptd
import parktools.tools as ptt
import pandas as pd

df = pd.read_csv('xBA/data/all_outcomes_2018.csv')

X, y, outcomes = ptd.pre_process(df)
print(X.head())
print(y.head())
print(outcomes.head())

print('\ntest calc_BABIP()')
babip_2018 = ptt.calc_BABIP('../data/batted_balls_2018.csv')
babip_2019 = ptt.calc_BABIP('../data/batted_balls_2019.csv')
print('Calculated BABIP for 2018: {:.3f}'.format(babip_2018))
print('Calculated BABIP for 2019: {:.3f}'.format(babip_2019))

print('\ntest get_batting_league()')
ao, bb = ptd.get_batting_league(
    '2019-07-01', '2019-07-02'
)
print('\nall outcomes')
print(ao.head())
print('\nbatted balls')
print(bb.head())

print('\ntest filter_batting_player()')
df = ptd.filter_batting_player(
    '../data/batted_balls_2019.csv', 'gordon', 'dee'
)
print(df.head())

print('\ntest get_batting_player()')
df, df0 = ptd.get_batting_player(
    '2019-07-01', '2019-07-02',
    'Gordon', 'Dee'
)
print(df.head())