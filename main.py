from model import EnergyModel
import pandas as pd
import dateutil

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
# 500 kW Scenario

KWARGS = {
        'battery_capacity': 534,
        'grid_limit': 500
    }
es_500 = EnergyModel(input_df=df500energy, capex=False, **KWARGS)
es_500.solve()
ans_500_energy_flow = es_500.flows
ans_500_cost = es_500.costs
####################################################################
# Sizing storage scenariio 500 kW limit
KWARGS = {

        'grid_limit': 500
    }
es_sizing = EnergyModel(input_df=df500energy, capex=False, **KWARGS)
es_sizing.solve()
ans_sizing500_energy_flow = es_sizing.flows
ans_sizing500_cost = es_sizing.costs
ans_sizing500_battery_cap = es_sizing.battery_cap

####################################################################
# Sizing storage scenario 400 kW limit
# KWARGS = {
#
#         'grid_limit': 400
#     }
# es_sizing = EnergyModel(input_df=df500energy, capex=False, **KWARGS)
# es_sizing.solve()
# ans_sizing400_energy_flow = es_sizing.flows
# ans_sizing400_cost = es_sizing.costs
# ans_sizing400_battery_cap = es_sizing.battery_cap

##############################
############################################################
# comparison of results
# pv_total_opt = r_dict['pv_self'] + r_dict['pv_curtailed']
# pv_total_opt = pv_total_opt.sum()
# pv_total_base = df500['uncurtailed_solar_power'].sum() * 0.25
# selfconsumption_pv_opt = r_dict['pv_self'].sum()
# selfconsumption_pv_base = (df500['uncurtailed_solar_power'] - df500['curtailed_power']).sum() * 0.25
# grid_base = df500['grid_power'].sum() * 0.25
# grid_opt = r_dict['grid'].sum()
