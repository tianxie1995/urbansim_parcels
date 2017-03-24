import os
import orca
from urbansim_parcels import models
from urbansim_parcels.utils import subset_orca_table
from sd_example import custom_models

if os.environ.get('TRAVIS') == 'true':
    # Subset tables in Travis build due to memory constraints
    subset_orca_table('households')
    subset_orca_table('jobs')
    subset_orca_table('buildings')

first_year = 2013
last_year = 2013
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
          "feasibility",  # supply/proforma models
          "residential_developer",
          "non_residential_developer",
          "model_integration_indicators",
          # "buildings_to_uc", #export buildings to urban canvas
          ], iter_vars=range(first_year,
                             last_year + 1))
