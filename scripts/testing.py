import pyxBA.tools as px

print('\ntest calc_BABIP()')
babip_2018 = px.calc_BABIP('../data/batted_balls_2018.csv')
babip_2019 = px.calc_BABIP('../data/batted_balls_2019.csv')
print('Calculated BABIP for 2018: {:.3f}'.format(babip_2018))
print('Calculated BABIP for 2019: {:.3f}'.format(babip_2019))

print('\ntest filer_player_batting_data()')
df = px.filter_player_batting_data(
    '../data/batted_balls_2019.csv', 'gordon', 'dee'
)
print(df.head())

print('\ntest get_league_batting_data()')
ao, bb = px.get_league_batting_data(
    '2019-07-01', '2019-07-02'
)
print('\nall outcomes')
print(ao.head())
print('\nbatted balls')
print(bb.head())
