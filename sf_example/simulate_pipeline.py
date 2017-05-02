from urbansim.utils import misc
from urbansim_parcels import models
from urbansim_parcels import utils
from sf_example import custom_models
import pandas as pd
import orca


orca.run([
    "rsh_simulate",  # residential sales hedonic
    "nrh_simulate",  # non-residential rent hedonic

    "households_relocation",  # households relocation model
    "hlcm_simulate",  # households location choice
    "simple_households_transition",  # households transition

    "jobs_relocation",  # jobs relocation model
    "elcm_simulate",  # employment location choice
    "simple_jobs_transition",  # jobs transition

    "feasibility_with_pipeline",  # supply/proforma models
    "residential_developer_pipeline",
    "non_residential_developer_pipeline",
    "build_from_pipeline"
], iter_vars=[2010, 2011, 2012])