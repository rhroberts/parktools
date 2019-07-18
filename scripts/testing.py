from parktools import tools, data
import pandas as pd

df = pd.read_csv('../data/all_outcomes_2018.csv')
df2 = pd.read_csv('../data/batted_balls_2018.csv')

X, y, outcomes = data.pre_process(df)
# print(X.head())
# print(y.head())
# print(outcomes.head())

# print('\ntest calc_BABIP()')
# babip_2018 = tools.calc_BABIP('../data/batted_balls_2018.csv')
# babip_2019 = tools.calc_BABIP('../data/batted_balls_2019.csv')
# print('Calculated BABIP for 2018: {:.3f}'.format(babip_2018))
# print('Calculated BABIP for 2019: {:.3f}'.format(babip_2019))


# print('\ntest get_batting_league()')
# ao, bb = data.get_batting_league(
#     '2019-07-01', '2019-07-02'
# )
# print('\nall outcomes')
# print(ao.head())
# print('\nbatted balls')
# print(bb.head())
# 
# print('\ntest filter_batting_player()')
# df = data.filter_batting_player(
#     '../data/batted_balls_2019.csv', 'gordon', 'dee'
# )
# print(df.head())
# 
# print('\ntest get_batting_player()')
# df, df0 = data.get_batting_player(
#     '2019-07-01', '2019-07-02',
#     'Gordon', 'Dee'
# )
# print(df.head())
