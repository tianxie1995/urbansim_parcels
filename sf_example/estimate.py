from urbansim_parcels import models, variables, datasources
import orca

steps = [
    'rsh_estimate',
    'nrh_estimate',
    'hlcm_estimate',
    'elcm_estimate'
]

orca.run(steps)