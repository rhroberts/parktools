from parktools.tools import calc_BA
import pandas as pd

df = pd.read_csv('xBA/data/all_outcomes_2018.csv', index_col=0)

print('\ntest calc_BA()')
ba_2018 = calc_BA(df, from_file=False)
print('Average BA for 2018: {:.3f}'.format(ba_2018))

print(df.loc[df['batter'] == 596019, 'events'].unique())
AB = df.groupby('batter', axis=0).size()
AB = AB.sort_values(ascending=False)
# AB = pd.DataFrame.from_dict(AB_count, orient='index')
print(AB)
