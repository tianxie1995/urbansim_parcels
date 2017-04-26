import os
import orca
from urbansim_parcels import models
from sd_example import custom_models


orca.run(["build_networks"])  # initialize network accessibility engine
orca.run([
          # "scheduled_development_events",
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
          "feasibility_with_pipeline",  # supply/proforma models
          "residential_developer_pipeline",
          "non_residential_developer_pipeline",
          "build_from_pipeline",
          "model_integration_indicators",
          # "buildings_to_uc", #export buildings to urban canvas
          ], iter_vars=[2012, 2013, 2014])
