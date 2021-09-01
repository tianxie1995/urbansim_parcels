# Urbansim Parcels change log 
##  0.1.dev1
- Replaces the `to_frame()` method for `local.copy()` for the `parcels` table
- Establishes two different criteria for the creation of the `feasibility` DataFrame if there are several proposals to keep in the SqFtProForma.
- Redesigns the `compute_units_to_build()` function, replacing the aggregate number of agents to allocate with a DataFrameWrapper with the agents information  and the number of units with the name of the type of units for those agents. This function can now return an integer or a data frame for the units to build, as wells as taking the target vacancy as float or as an array.
