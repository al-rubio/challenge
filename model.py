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

ENERGY_COST = 0.22  # EUR/kWh
DEMAND_CHARGE = 200  # EUR/kWh
PV_COST = 0.07  # EUR/kWh

def power_balance_check(df):
    """

    :param df: dataframe with power series 15 min resol
    :return: total load imbalance energy (positive=shortage of generation), df with imbalance 15 min resol
    """
    gen = df[['uncurtailed_solar_power', 'grid_power']].sum(axis=1) - df['curtailed_power'] + df['ess_power']
    load = df['load']
    power_imbalance = (load - gen).round(3)

    a = df[(df['load'] > gen)]
    return power_imbalance.reindex(a.index).sum() * 0.25, power_imbalance


def cost_calc(grid_total, grid_max, pv_self):
    return ENERGY_COST * grid_total + DEMAND_CHARGE * grid_max + PV_COST * pv_self


class EnergyModel:
    def __init__(self, input_df, capex=False, **kwargs):
        self._energy_cost = kwargs.get('energy_cost', 0.22)
        self._grid_limit = kwargs.get('grid_limit', 485)
        self._battery_cap = kwargs.get('battery_capacity', 534)
        wacc = kwargs.get('wacc', 0.05)
        project_life = kwargs.get('project_life', 10)
        battery_capex = kwargs.get('battery_capex', 417.45)
        battery_max_cap = kwargs.get('battery_max_cap', 1000)
        self._input_data = input_df
        df_index = input_df.index
        self._model = None
        self._energy_system = EnergySystem(timeindex=df_index)
        self.costs = None
        self.flows = None
        # Build energy system
        # main buses
        b_dist = Bus(label='b_dist')
        b_exp = Bus(label='b_exp')
        b_prod = Bus(label='b_prod', outputs={b_dist: Flow(variable_costs=PV_COST), b_exp: Flow()})
        s_exp = Sink(label='s_exp', inputs={b_exp: Flow()})
        self._energy_system.add(b_dist, b_prod, b_exp, s_exp)

        # pv source
        s_pv = Source(label='s_pv',
                      outputs={b_prod: Flow(fix=input_df['pv'].values, nominal_value=1)})
        self._energy_system.add(s_pv)

        # battery storage
        if 'battery_capacity' in kwargs:
            self.battery_cap = kwargs.get('battery_capacity')
            sto_battery = GenericStorage(label='sto_battery',
                                         inputs={b_dist: Flow(max=400, nominal_value=1)},
                                         outputs={b_dist: Flow(max=400, nominal_value=1)},
                                         nominal_storage_capacity=self.battery_cap, balanced=True,
                                         loss_rate=0.00, initial_storage_level=0.5,  # Todo: add energy loss
                                         inflow_conversion_factor=1, outflow_conversion_factor=1)
        else:
            epc_battery = economics.annuity(capex=battery_capex, n=project_life, wacc=wacc)
            sto_battery = GenericStorage(label='sto_battery',
                                         inputs={b_dist: Flow(max=400, nominal_value=1)},
                                         outputs={b_dist: Flow(max=400, nominal_value=1)}, balanced=True,
                                         investment=Investment(ep_costs=epc_battery, maximum=battery_max_cap),
                                         loss_rate=0.00, initial_storage_level=0.5,  # Todo: add energy loss
                                         inflow_conversion_factor=1, outflow_conversion_factor=1)

        self._energy_system.add(sto_battery)

        # external markets
        m_grid = Source(label='m_grid',
                        outputs={b_dist: Flow(variable_costs=ENERGY_COST, max=self._grid_limit, nominal_value=1)})
        load = Sink(label='load',
                    inputs={b_dist: Flow(fix=input_df['load'].values, nominal_value=1)})
        self._energy_system.add(m_grid, load)

        # dispatch optimization
    def solve(self):
        self._model = Model(self._energy_system)
        self._model.solve(solver='cbc')

        r_dict = dict()
        results = processing.results(self._model)
        results_keys = views.convert_keys_to_strings(results)
        try:
            self.battery_cap = results_keys[('sto_battery', 'None')]['scalars']['invest']
        except:
            pass
        r_dict['pv_self'] = results_keys[('b_prod', 'b_dist')]['sequences']
        r_dict['pv_curtailed'] = results_keys[('b_prod', 'b_exp')]['sequences']
        r_dict['battery'] = results_keys[('sto_battery', 'b_dist')]['sequences'] - \
                            results_keys[('b_dist', 'sto_battery')]['sequences']
        r_dict['battery_soc'] = results_keys[('sto_battery', 'None')]['sequences'] / self.battery_cap
        r_dict['grid'] = results_keys[('m_grid', 'b_dist')]['sequences']
        r_df = pd.concat(r_dict.values(), axis=1, sort=False)
        r_df.columns = r_dict.keys()
        self.costs = cost_calc(grid_total=r_df['grid'].sum(),
                               grid_max=r_df['grid'].max(),
                               pv_self=r_df['pv_self'].sum())
        self.flows = r_df


if __name__ == '__main__':

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
    df500energy = df500[['load', 'ess_power', 'grid_power', 'uncurtailed_solar_power']].resample('1h').sum() * 0.25
    df500energy.columns = ['load', 'battery', 'grid', 'pv']
    es = EnergyModel(input_df=df500energy, capex=False, **KWARGS)
    es.solve()
    result = es.results()