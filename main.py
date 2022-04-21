from model import EnergyModel, cost_calc, power_balance_check, DEMAND_CHARGE, PV_COST, ENERGY_COST
import pandas as pd
import dateutil
import datetime
import matplotlib.pyplot as plt
import numpy as np

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
a_pvself400 = (df400['uncurtailed_solar_power']-df400['curtailed_power'])
a_sc400cost = cost_calc(grid_total=a_grid400.sum()*0.25,
                        grid_max=a_gridmax400,
                        pv_self=a_pvself400.sum()*0.25)
# 500 Scenario
a_grid500 = df_scn500_imbalance + df500['grid_power']
a_gridmax500 = a_grid500.max()
a_pvself500 = (df500['uncurtailed_solar_power']-df500['curtailed_power'])
a_sc500cost = cost_calc(grid_total=a_grid500.sum()*0.25,
                        grid_max=a_gridmax500,
                        pv_self=a_pvself500.sum()*0.25)
######################################################################################
######################################################################################
# plots
LINE_WIDTH = 2.5
#######################################################
# Cost Share
fig0_a, ax0_a = plt.subplots(2)
cgrid400 = a_grid400.sum() * 0.25 * ENERGY_COST
cpv400 = a_pvself400.sum()*0.25
cpmax400 = a_gridmax400 * DEMAND_CHARGE

cgrid500 = a_grid500.sum() * 0.25 * ENERGY_COST
cpv500 = a_pvself500.sum()*0.25
cpmax500 = a_gridmax500 * DEMAND_CHARGE

pie_data_400 = (['From grid', 'From PV', 'From max demand'],
                [cgrid400, cpv400, cpmax400])
pie_data_500 = (['From grid', 'From PV', 'From max demand'],
                [cgrid500, cpv500, cpmax500])
ax0_a[0].pie(pie_data_400[1],
          labels=pie_data_400[0],
          autopct='%1.1f%%',)
ax0_a[0].set_title('Case A: Cost share Scenario 400')
ax0_a[1].pie(pie_data_500[1],
          labels=pie_data_500[0],
          autopct='%1.1f%%',)
ax0_a[1].set_title('Case A: Cost share Scenario 500')


#######################################################
# Grid import
fig_a, ax_a = plt.subplots(1)
y400 = a_grid400
y500 = a_grid500
a = ax_a.plot(y400, label='Grid import 400kW Scenario', color='blue',
              linewidth=LINE_WIDTH)
b = ax_a.plot(y500, label='Grid import 500kW Scenario', color='red',
              linewidth=LINE_WIDTH)
ax_a.set_xlabel('Date')
ax_a.set_ylabel('Power (kW)')
ax_a.grid(True)
ax_a.set_title('Case A: From given data')
ax_a.legend()
# Pie charts
fig1_a, ax1_a = plt.subplots(2)
pie_data_400 = (['From grid', 'From PV'],
                [a_grid400.sum(), a_pvself400.sum()])
pie_data_500 = (['From grid', 'From PV'],
                [a_grid500.sum(), a_pvself500.sum()])
ax1_a[0].pie(pie_data_400[1],
          labels=pie_data_400[0],
          autopct='%1.1f%%',)
ax1_a[0].set_title('Case A: Load Supply Scenario 400')


ax1_a[1].pie(pie_data_500[1],
          labels=pie_data_500[0],
          autopct='%1.1f%%',)
ax1_a[1].set_title('Case A: Load Supply Scenario 500')

##################################
# Battery charge
fig2_a, ax2_a = plt.subplots(2)
ypv = a_pvself500
yload = df400['load']
y500_grid = -df500[(df500['ess_power'] < 0) &
                   (df500['uncurtailed_solar_power'] == 0)]['ess_power'].reindex(yload.index).replace(np.nan, 0).sum() \
            * 0.25
y500_pv = -df500[(df500['ess_power'] < 0) &
                 (df500['uncurtailed_solar_power'] > 0)]['ess_power'].reindex(yload.index).replace(np.nan, 0).sum() \
            * 0.25
y400_grid = -df400[(df400['ess_power'] < 0) &
                   (df400['uncurtailed_solar_power'] == 0)]['ess_power'].reindex(yload.index).replace(np.nan, 0) .sum() \
            * 0.25
y400_pv = -df500[(df400['ess_power'] < 0) &
                 (df400['uncurtailed_solar_power'] > 0)]['ess_power'].reindex(yload.index).replace(np.nan, 0).sum() \
            * 0.25

#pie charts
ax2_a[0].pie([y400_grid, y400_pv],
          labels=['From grid', 'From PV'],
          autopct='%1.1f%%',)
ax2_a[0].set_title('Case A: Battery charging 400 Scenario')
ax2_a[1].pie([y500_grid, y500_pv],
          labels=['From grid', 'From PV'],
          autopct='%1.1f%%',)
ax2_a[1].set_title('Case A: Battery charging 500 Scenario')

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

# Pie charts
fig1_b, ax1_b = plt.subplots(2)
pie_data_400 = (['From grid', 'From PV'],
                [b_400_energy_flow['grid'].sum(), b_400_energy_flow['pv_self'].sum()])
pie_data_500 = (['From grid', 'From PV'],
                [b_500_energy_flow['grid'].sum(), b_500_energy_flow['pv_self'].sum()])
ax1_b[0].pie(pie_data_400[1],
          labels=pie_data_400[0],
          autopct='%1.1f%%',)
ax1_b[0].set_title('Case B: Load Supply Scenario 400')


ax1_b[1].pie(pie_data_500[1],
          labels=pie_data_500[0],
          autopct='%1.1f%%',)
ax1_b[1].set_title('Case B: Load Supply Scenario 500')

########################################################################
#########################################################################
# Sizing storage scenario 485 kW limit
KWARGS = {

        'grid_limit': 485
    }
es_sizing = EnergyModel(input_df=df500energy, capex=False, **KWARGS)
es_sizing.solve()
c_sizing400_energy_flow = es_sizing.flows
c_sizing400_cost = es_sizing.costs
c_sizing400_battery_cap = es_sizing.battery_cap

####################################################################
# Sizing storage scenariio 500 kW limit
KWARGS = {

        'grid_limit': 500
    }
es_sizing = EnergyModel(input_df=df500energy, capex=False, **KWARGS)
es_sizing.solve()
c_sizing500_energy_flow = es_sizing.flows
d_sizing500_cost = es_sizing.costs
d_sizing500_battery_cap = es_sizing.battery_cap

####################################################################
