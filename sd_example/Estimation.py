
from urbansim_parcels import models
from sd_example import custom_models
import orca


# In[2]:

orca.run(["build_networks", "neighborhood_vars"])


# ## Residential Price Hedonice Estimation

# In[22]:

orca.run(["rsh_estimate"])


# In[23]:

orca.run(["rsh_simulate"])


# ## Non-residential Rent Hedonic Estimation

# In[18]:

orca.run(["nrh_estimate"])


# In[6]:

orca.run(["nrh_simulate"])


# In[19]:

orca.run(["nrh_estimate2"])


# In[8]:

orca.run(["nrh_simulate2"])


# ## ELCM Estimation

# In[25]:

orca.run(["nrh_simulate"])
orca.run(["nrh_simulate2"])
orca.run(["elcm_estimate"])


# In[26]:

orca.run(["jobs_transition", "elcm_simulate"])


# ## HLCM Estimation

# In[26]:

orca.run(["rsh_simulate"])
orca.run(["hlcm_estimate"])


# In[24]:

orca.run(["households_transition_basic"], iter_vars = [2013])
orca.run(["hlcm_simulate"])

