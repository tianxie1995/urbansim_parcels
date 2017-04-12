import os
import orca
from urbansim_parcels import models
from sd_example import custom_models


start_year = 2013
orca.add_injectable('start_year', start_year)
last_year = 2018


orca.run(["build_networks"])  # initialize network accessibility engine
orca.run(["scheduled_development_events",
          "neighborhood_vars",  # accessibility variables
          "rsh_simulate",
          "nrh_simulate",
          "nrh_simulate2",
          "jobs_transition",
          "jobs_relocation",
          "elcm_simulate",
          "households_transition_basic",
          "households_relocation",
          "hlcm_simulate",  # demand/location models
          "price_vars",
          "regional_occupancy",
          "feasibility_with_occupancy",  # supply/proforma models
          "residential_developer_profit",
          "non_residential_developer_profit",
          "model_integration_indicators",
          # "buildings_to_uc", #export buildings to urban canvas
          ], iter_vars=range(start_year,
                             last_year + 1))
