df_index = df400energy.index
energy_system = EnergySystem(timeindex=df_index)

# main buses
b_dist = Bus(label='b_dist')
b_exp = Bus(label='b_exp')
b_prod = Bus(label='b_prod', outputs={b_dist: Flow(), b_exp: Flow()})
s_exp = Sink(label='s_exp', inputs={b_exp: Flow()})
energy_system.add(b_dist, b_prod, b_exp, s_exp)

# pv source
s_pv = Source(label='s_pv',
              outputs={b_prod: Flow(fix=df400energy['pv'].values, nominal_value=1)})
energy_system.add(s_pv)

# battery storage
sto_battery = GenericStorage(label='sto_battery',
                             inputs={b_dist: Flow(max=400, nominal_value=1)},
                             outputs={b_dist: Flow(max=400, nominal_value=1)},
                             nominal_storage_capacity=534,
                             loss_rate=0.00, initial_storage_level=0,  # Todo: add energy loss
                             inflow_conversion_factor=1, outflow_conversion_factor=1)

# battery_in = df400energy[df400energy['battery'] < 0]['battery'].reindex(df_index).replace(np.nan, 0)
# battery_out = df400energy[df400energy['battery'] > 0]['battery'].reindex(df_index).replace(np.nan, 0)
# sto_battery = GenericStorage(label='sto_battery',
#                              inputs={b_dist: Flow(fix=battery_in,
#                                                   nominal_value=1)},
#                              outputs={b_dist: Flow(fix=battery_out,
#                                                    nominal_value=1)},
#                              nominal_storage_capacity=534,
#                              loss_rate=0.00, initial_storage_level=0,  # Todo: add energy loss
#                              inflow_conversion_factor=1, outflow_conversion_factor=1)
energy_system.add(sto_battery)

# external markets
m_grid = Source(label='m_grid',
                outputs={b_dist: Flow(variable_costs=ENERGY_COST, max=GRID_LIMIT, nominal_value=1)})
load = Sink(label='load',
            inputs={b_dist: Flow(fix=df400energy['load'].values, nominal_value=1)})
energy_system.add(m_grid, load)

import pandas as pd
import dateutil
import numpy as np
from oemof.solph import (Bus, Sink, Source, EnergySystem, Model, Investment, GenericStorage,
                         Flow, processing, views)
from oemof.tools import economics

df400 = pd.read_csv('Results_400limit.csv', sep=",",
                    index_col='index',
                    parse_dates=['index'],
                    date_parser=dateutil.parser.parse)
power_balance_400 = np.round(df400[['ess_power', 'grid_power', 'uncurtailed_solar_power']].sum(axis=1) - \
                    df400[['load', 'curtailed_power']].sum(axis=1), 3)  # power excess, should be equal to curtailment
pb400filter = power_balance_400[power_balance_400 != 0]
pgrid_400_exceed = df400[df400['grid_power'] > 400]
df_unbalance_400 = df400.loc[pb400filter.index]

####################################################################
df500 = pd.read_csv('Results_500limit.csv', sep=",",
                    index_col='index',
                    parse_dates=['index'],
                    date_parser=dateutil.parser.parse)

power_balance_500 = np.round(df500[['ess_power', 'grid_power', 'uncurtailed_solar_power']].sum(axis=1) - \
                    df500[['load', 'curtailed_power']].sum(axis=1), 3)  # power excess, should be equal to curtailment
pb500filter = power_balance_500[power_balance_500 != 0]
pgrid_500_exceed = df500[df500['grid_power'] > 500]


#####################################################################
#####################################################################

# Optimization of 400 kW Scenario
ENERGY_COST = 0.22  # EUR/kWh
DEMAND_CHARGE = 200  # EUR/kWh
PV_COST = 0.07  # EUR/kWh LCOE
GRID_LIMIT = 482.5  # kW grid limit 400 kW/ 400 kWh in 1h time steps

df400energy = df400[['load', 'ess_power', 'grid_power', 'uncurtailed_solar_power']].resample('1h').sum() * 0.25
df400energy.columns = ['load', 'battery', 'grid', 'pv']

# dispatch optimization

model = Model(energy_system)
model.solve(solver='cbc')

# results
try:
    r_dict = dict()
    results = processing.results(model)
    results_keys = views.convert_keys_to_strings(results)

    r_dict['pv_self'] = results_keys[('b_prod', 'b_dist')]['sequences']
    r_dict['pv_curtailed'] = results_keys[('b_prod', 'b_exp')]['sequences']
    r_dict['battery'] = results_keys[('sto_battery', 'b_dist')]['sequences'] - \
                        results_keys[('b_dist', 'sto_battery')]['sequences']
    r_dict['grid'] = results_keys[('m_grid', 'b_dist')]['sequences']

    ############################################################
    # comparison of results
    pv_total_opt = r_dict['pv_self'] + r_dict['pv_curtailed']
    pv_total_opt = pv_total_opt.sum()
    pv_total_base = df500['uncurtailed_solar_power'].sum() * 0.25
    selfconsumption_pv_opt = r_dict['pv_self'].sum()
    selfconsumption_pv_base = (df500['uncurtailed_solar_power'] - df500['curtailed_power']).sum() * 0.25
    grid_base = df500['grid_power'].sum() * 0.25
    grid_opt = r_dict['grid'].sum()
except:
    pass
