from dataclasses import dataclass, field
from ISST import Risk, RiskTable

import pymc as pm
import numpy as np
import pandas as pd

import os
from pathlib import Path
@dataclass
class DesignSystem:
    name: str = field(init=True)

    risks: list[Risk] = field(init=True)

    model_context: pm.Model = field(init=True)

    # System-Wide Schedule Risk Table
    schedule_risk_table: RiskTable = field(init=True)

    # System-Wide Cost Risk Table
    cost_risk_table: RiskTable = field(init=True)

    # System-Wide Technical Risk Tables
    technical_risk_tables: list[RiskTable] = field(init=True)

    def __post_init__(self):
        assert self.schedule_risk_table is not None
        assert self.cost_risk_table is not None
        assert self.technical_risk_tables is not None

        self.schedule_risk_levels = np.zeros(np.asarray(self.schedule_risk_table.utility_breakpoints).shape[0])
        self.cost_risk_levels = np.zeros(np.asarray(self.cost_risk_table.utility_breakpoints).shape[0])

        self.max_tech_risk_sizes = np.zeros((len(self.technical_risk_tables)))
        mtrs = 0
        for ii, risk_table in enumerate(self.technical_risk_tables):
            mtrs = max(mtrs, np.asarray(risk_table.utility_breakpoints).shape[0])
            self.max_tech_risk_sizes[ii] = mtrs

    def generate_system_specification(self):

        rootpath = os.getcwd()
        system_path = Path(rootpath, self.name)
        os.makedirs(system_path, exist_ok=True)

        schedule_df = pd.DataFrame(data={'Minimum Schedule Impact': np.zeros(len(self.risks)),
                                         'Maximum Schedule Impact': np.zeros(len(self.risks)),
                                         'Most Likely Schedule Impact': np.zeros(len(self.risks))},
                                   index=[risk.name for risk in self.risks])

        with open(Path(system_path, f'{self.name} Schedule Risks.csv'), 'w') as f:
            schedule_df.to_csv(f, index=True, header=True)

        cost_df = pd.DataFrame(data={'Minimum Cost Impact': np.zeros(len(self.risks)),
                                     'Maximum Cost Impact': np.zeros(len(self.risks)),
                                     'Most Likely Cost Impact': np.zeros(len(self.risks))},
                               index=[risk.name for risk in self.risks])

        with open(Path(system_path, f'{self.name} Cost Risks.csv'), 'w') as f:
            cost_df.to_csv(f, index=True, header=True)

        for tech_risk in self.technical_risk_tables:
            tech_risk_df = pd.DataFrame(data={f'Minimum {tech_risk.name} Impact': np.zeros(len(self.risks)),
                                              f'Maximum {tech_risk.name} Impact': np.zeros(len(self.risks)),
                                              f'Most Likely {tech_risk.name} Impact': np.zeros(len(self.risks))},
                                        index=[risk.name for risk in self.risks])

            with open(Path(system_path, f'{self.name} {tech_risk.name} Risks.csv'), 'w') as f:
                tech_risk_df.to_csv(f, index=True, header=True)

        return

    def read_system_specification(self):

        rootpath = os.getcwd()
        system_path = Path(rootpath, self.name)

        with open(Path(system_path, f'{self.name} Schedule Risks.csv'), 'r') as f:
            schedule_df = pd.read_csv(f)

        with open(Path(system_path, f'{self.name} Cost Risks.csv'), 'r') as f:
            cost_df = pd.read_csv(f)

        for risk in self.risks:
            risk.schedule_risk_minimum_value = schedule_df.loc[risk.name, 'Minimum Schedule Impact']
            risk.schedule_risk_maximum_value = schedule_df.loc[risk.name, 'Maximum Schedule Impact']
            risk.schedule_risk_most_likely_value = schedule_df.loc[risk.name, 'Most Likely Schedule Impact']

            risk.cost_risk_minimum_value = cost_df.loc[risk.name, 'Minimum Cost Impact']
            risk.cost_risk_maximum_value = cost_df.loc[risk.name, 'Maximum Cost Impact']
            risk.cost_risk_most_likely_value = cost_df.loc[risk.name, 'Most Likely Cost Impact']

        for tech_risk in self.technical_risk_tables:

            with open(Path(system_path, f'{self.name} {tech_risk.name} Risks.csv'), 'r') as f:
                tech_risk_df = pd.read_csv(f)

            for risk in self.risks:
                risk.technical_risk_minimum_values.append(
                    tech_risk_df.loc[risk.name, f'Minimum {tech_risk.name} Impact'])
                risk.technical_risk_maximum_values.append(
                    tech_risk_df.loc[risk.name, f'Maximum {tech_risk.name} Impact'])
                risk.technical_risk_most_likely_values.append(
                    tech_risk_df.loc[risk.name, f'Most Likely {tech_risk.name} Impact'])

        return

    def analyze_system(self):

        risk_schedule_mins = []
        risk_schedule_maxs = []
        risk_schedule_mls = []

        risk_cost_mins = []
        risk_cost_maxs = []
        risk_cost_mls = []

        for risk in self.risks:
            risk_schedule_mins.append(risk.schedule_risk_minimum_value)
            risk_schedule_maxs.append(risk.schedule_risk_maximum_value)
            risk_schedule_mls.append(risk.schedule_risk_most_likely_value)

            risk_cost_mins.append(risk.cost_risk_minimum_value)
            risk_cost_maxs.append(risk.cost_risk_maximum_value)
            risk_cost_mls.append(risk.cost_risk_most_likely_value)

        s_min = min(risk_schedule_mins)
        s_max = max(risk_schedule_maxs)
        s_rng = s_max - s_min

        s_min_arr = (np.asarray(risk_schedule_mins) - s_min) / s_rng
        s_max_arr = (np.asarray(risk_schedule_maxs) - s_min) / s_rng
        s_ml_arr = (np.asarray(risk_schedule_mls) - s_min) / s_rng

        s_mu = (s_min_arr + 4 * s_ml_arr + s_max_arr) / 6
        s_sigma = np.sqrt((s_mu - s_min_arr) * (s_max_arr - s_mu) / 7)

        c_min = min(risk_cost_mins)
        c_max = max(risk_cost_maxs)
        c_rng = c_max - c_min

        c_min_arr = (np.asarray(risk_cost_mins) - c_min) / c_rng
        c_max_arr = (np.asarray(risk_cost_maxs) - c_min) / c_rng
        c_ml_arr = (np.asarray(risk_cost_mls) - c_min) / c_rng

        c_mu = (c_min_arr + 4 * c_ml_arr + c_max_arr) / 6
        c_sigma = np.sqrt((c_mu - c_min_arr) * (c_max - c_mu_arr) / 7)

        tech_risk_mins = np.zeros((len(self.risks), len(self.technical_risk_tables)))
        tech_risk_maxs = np.zeros((len(self.risks), len(self.technical_risk_tables)))
        tech_risk_mls = np.zeros((len(self.risks), len(self.technical_risk_tables)))

        t_mins = np.zeros(len(self.technical_risk_tables))
        t_maxs = np.zeros(len(self.technical_risk_tables))
        t_mls = np.zeros(len(self.technical_risk_tables))

        t_mus = np.zeros(len(self.technical_risk_tables))
        t_sigmas = np.zeros(len(self.technical_risk_tables))

        for jj, tech_risk in enumerate(self.technical_risk_tables):
            for ii, risk in enumerate(self.risks):
                tech_risk_mins[ii, jj] = risk.technical_risk_minimum_values[jj]
                tech_risk_maxs[ii, jj] = risk.technical_risk_maximum_values[jj]
                tech_risk_mls[ii, jj] = risk.technical_risk_most_likely_values[jj]

            t_mins[jj] = min(tech_risk_mins[:, jj])
            t_maxs[jj] = max(tech_risk_maxs[:, jj])
            t_mls[jj] = max(tech_risk_mls[:, jj])

            t_rngs = t_maxs - t_mins

            t_mins_scaled = (np.asarray(tech_risk_mins[:, jj]) - t_mins[jj]) / t_rngs[jj]
            t_maxs_scaled = (np.asarray(tech_risk_maxs[:, jj]) - t_mins[jj]) / t_rngs[jj]
            t_mls_scaled = (np.asarray(tech_risk_mls[:, jj]) - t_mins[jj]) / t_rngs[jj]

            t_mus[jj] = (t_mins_scaled[jj] + 4 * t_mls_scaled[jj] + t_maxs_scaled[jj]) / 6
            t_sigmas[jj] = np.sqrt((t_mus[jj] - t_mins_scaled[jj]) * (t_maxs_scaled[jj] - t_mus[jj]) / 7)

        with self.model_context:
            schedule_vars = pm.Deterministic('Schedule',
                                             pm.Beta('Schedule_Scaled', mu=s_mu, sigma=s_sigma,
                                                     shape=len(self.risks)) * s_rng + s_min)
            cost_vars = pm.Deterministic('Cost',
                                         pm.Beta('Cost_Scaled', mu=c_mu, sigma=c_sigma,
                                                 shape=len(self.risks)) * c_rng + c_min)
            technical_vars = pm.Deterministic(
                pm.Beta('Technical_Scaled', mu=t_mus, sigma=t_sigmas,
                        shape=(len(self.risks), len(self.technical_risk_tables))) * t_rngs + t_mins)

            risk_priors = []
            for ii, risk in enumerate(self.risks):
                risk_priors.append(risk.basline_likelihood)

            total_schedule_impact = pm.math.dot(schedule_vars, np.asarry(risk_priors))
            total_cost_impact = pm.math.dot(cost_vars, np.asarry(risk_priors))
            total_technical_impact = pm.math.dot(technical_vars.T, np.asarry(risk_priors)).T

            idata = pm.sample()

        return idata