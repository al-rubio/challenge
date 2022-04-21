from model import EnergyModel, cost_calc, power_balance_check
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

# Check power balance of input data

scn400_imbalance, df_scn400_imbalance = power_balance_check(df400)
scn500_imbalance, df_scn500_imbalance = power_balance_check(df500)

########################################################################################
# Cost Calculation with provided input data
# 400 Scenario
a_grid400 = df_scn400_imbalance + df400['grid_power']
a_gridmax400 = a_grid400.max()
a_sc400cost = cost_calc(grid_total=a_grid400.sum()*0.25,
                        grid_max=a_gridmax400,
                        pv_self=(df400['uncurtailed_solar_power']-df400['curtailed_power']).sum()*0.25)
# 500 Scenario
a_grid500 = df_scn500_imbalance + df500['grid_power']
a_gridmax500 = a_grid500.max()
a_sc500cost = cost_calc(grid_total=a_grid500.sum()*0.25,
                        grid_max=a_gridmax500,
                        pv_self=(df500['uncurtailed_solar_power']-df500['curtailed_power']).sum()*0.25)

##########################################################################################
##########################################################################################
# Dispatch optimization
# In the provided input data, batteries are charged from the grid and 400 kW power grid import is not
# kept
df400energy = df400[['load', 'ess_power', 'grid_power', 'uncurtailed_solar_power']].resample('1h').sum() * 0.25
df400energy.columns = ['load', 'battery', 'grid', 'pv']

df500energy = df500[['load', 'ess_power', 'grid_power', 'uncurtailed_solar_power']].resample('1h').sum() * 0.25
df500energy.columns = ['load', 'battery', 'grid', 'pv']
diff = (df400energy - df500energy).sum()
#################################################################
# 400 kW Scenario
# Grid limits below 485 kW led to an unfeasible problem
KWARGS = {
        'battery_capacity': 534,
        'grid_limit': 485
    }
es_400 = EnergyModel(input_df=df500energy, capex=False, **KWARGS)
es_400.solve()
b_400_energy_flow = es_400.flows
b_400_cost = es_400.costs
#################################################################
# 500 kW Scenario

KWARGS = {
        'battery_capacity': 534,
        'grid_limit': 500
    }
es_500 = EnergyModel(input_df=df500energy, capex=False, **KWARGS)
es_500.solve()
b_500_energy_flow = es_500.flows
b_500_cost = es_500.costs
########################################################################
#########################################################################
# Sizing storage scenariio 500 kW limit
KWARGS = {

        'grid_limit': 500
    }
es_sizing = EnergyModel(input_df=df500energy, capex=False, **KWARGS)
es_sizing.solve()
ans_sizing500_energy_flow = es_sizing.flows
c_sizing500_cost = es_sizing.costs
c_sizing500_battery_cap = es_sizing.battery_cap

####################################################################
