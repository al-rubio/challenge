from model import EnergyModel
import pandas as pd
import dateutil

KWARGS = {
        'battery_capacity': 534,
        'grid_limit': 500
    }

df400 = pd.read_csv('Results_400limit.csv', sep=",",
                    index_col='index',
                    parse_dates=['index'],
                    date_parser=dateutil.parser.parse)
df500 = pd.read_csv('Results_500limit.csv', sep=",",
                    index_col='index',
                    parse_dates=['index'],
                    date_parser=dateutil.parser.parse)

df400energy = df400[['load', 'ess_power', 'grid_power', 'uncurtailed_solar_power']].resample('1h').sum() * 0.25
df400energy.columns = ['load', 'battery', 'grid', 'pv']

df500energy = df500[['load', 'ess_power', 'grid_power', 'uncurtailed_solar_power']].resample('1h').sum() * 0.25
df500energy.columns = ['load', 'battery', 'grid', 'pv']
diff = (df400energy - df500energy).sum()

#################################################################
#
es = EnergyModel(input_df=df500energy, capex=False, **KWARGS)
es.solve()
result = es.results()