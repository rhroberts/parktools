# get batted ball data for individual players
from pybaseball import playerid_lookup

player = playerid_lookup('gordon', 'dee')

all_outcomes_2019 = pd.read_csv('data/all_outcomes_2019.csv')
dee_gordon = all_outcomes_2019[all_outcomes_2019['batter'] == 543829]
dee_gordon.reset_index(inplace=True, drop=True)
dee_gordon.to_csv('data/dee_gordon_2019.csv', index=False)
dee_gordon.head()
dee_gordon['events'].value_counts()

player = playerid_lookup('encarnacion', 'edwin')
edwin = all_outcomes_2019[all_outcomes_2019['batter'] == 429665]
edwin.reset_index(inplace=True, drop=True)
edwin.to_csv('data/edwin_encarnacion.csv', index=False)
