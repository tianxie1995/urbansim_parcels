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

    "regional_absorption",
    "feasibility",  # compute development feasibility
    "residential_developer",  # build residential buildings
    "non_residential_developer",  # build non-residential buildings
], iter_vars=[2010])
