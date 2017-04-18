from urbansim.utils import misc
from urbansim_parcels import models
from urbansim_parcels import utils
from sf_example import custom_models
import pandas as pd
import orca


start_year = 2010
orca.add_injectable('start_year', start_year)
end_year = 2012

orca.run([
    "rsh_simulate",  # residential sales hedonic
    "nrh_simulate",  # non-residential rent hedonic

    "households_relocation",  # households relocation model
    "hlcm_simulate",  # households location choice
    "simple_households_transition",  # households transition

    "jobs_relocation",  # jobs relocation model
    "elcm_simulate",  # employment location choice
    "simple_jobs_transition",  # jobs transition

    "regional_occupancy",
    "feasibility_with_occupancy",  # compute development feasibility
    "residential_developer_profit",  # build residential buildings
    "non_residential_developer_profit",  # build non-residential buildings
], iter_vars=range(start_year, end_year + 1))
