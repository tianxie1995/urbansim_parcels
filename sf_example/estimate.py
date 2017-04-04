from urbansim_parcels import models
from urbansim_parcels import variables
from urbansim_parcels import datasources
from sf_example import custom_models
from sf_example import custom_variables
from sf_example import custom_datasources
import orca

steps = [
    'rsh_estimate',
    'nrh_estimate',
    'hlcm_estimate',
    'elcm_estimate'
]

orca.run(steps)
