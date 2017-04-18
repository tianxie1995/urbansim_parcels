from __future__ import print_function, division, absolute_import

import numpy as np
import orca
import pandas as pd
from urbansim.utils import misc

from urbansim_parcels import datasources
from urbansim_parcels import utils
from sf_example import custom_datasources


#####################
# ZONES VARIABLES
#####################


@orca.column('zones', 'sum_residential_units')
def sum_residential_units(buildings):
    return (buildings.residential_units
            .groupby(buildings.zone_id).sum().apply(np.log1p))


@orca.column('zones', 'sum_job_spaces')
def sum_nonresidential_units(buildings):
    return (buildings.job_spaces
            .groupby(buildings.zone_id).sum().apply(np.log1p))


@orca.column('zones', 'population')
def population(households, zones):
    s = households.persons.groupby(households.zone_id).sum().apply(np.log1p)
    return s.reindex(zones.index).fillna(0)


@orca.column('zones', 'jobs')
def jobs(jobs):
    return jobs.zone_id.groupby(jobs.zone_id).size().apply(np.log1p)


@orca.column('zones', 'ave_lot_sqft')
def ave_lot_sqft(buildings, zones):
    s = (buildings.unit_lot_size
         .groupby(buildings.zone_id).quantile().apply(np.log1p))
    return s.reindex(zones.index).fillna(s.quantile())


@orca.column('zones', 'ave_income')
def ave_income(households, zones):
    s = (households.income
         .groupby(households.zone_id).quantile().apply(np.log1p))
    return s.reindex(zones.index).fillna(s.quantile())


@orca.column('zones', 'hhsize')
def hhsize(households, zones):
    s = (households.persons
         .groupby(households.zone_id).quantile().apply(np.log1p))
    return s.reindex(zones.index).fillna(s.quantile())


@orca.column('zones', 'ave_unit_sqft')
def ave_unit_sqft(buildings, zones):
    s = (buildings.unit_sqft[buildings.general_type == "Residential"]
         .groupby(buildings.zone_id).quantile().apply(np.log1p))
    return s.reindex(zones.index).fillna(s.quantile())


@orca.column('zones', 'sfdu')
def sfdu(buildings, zones):
    s = (buildings.residential_units[buildings.building_type_id == 1]
         .groupby(buildings.zone_id).sum().apply(np.log1p))
    return s.reindex(zones.index).fillna(0)


@orca.column('zones', 'poor')
def poor(households, zones):
    s = (households.persons[households.income < 40000]
         .groupby(households.zone_id).sum().apply(np.log1p))
    return s.reindex(zones.index).fillna(0)


@orca.column('zones', 'renters')
def renters(households, zones):
    s = (households.persons[households.tenure == 2]
         .groupby(households.zone_id).sum().apply(np.log1p))
    return s.reindex(zones.index).fillna(0)


@orca.column('zones', 'zone_id')
def zone_id(zones):
    return zones.index


@orca.column('zones_prices', 'residential')
def residential(buildings, year):
    price_filter = ((buildings.general_type == "Residential")
                    & (buildings.year_built >= year - 5))
    return (buildings
            .residential_sales_price[price_filter]
            .groupby(buildings.zone_id).quantile())


@orca.column('zones_prices', 'retail')
def retail(buildings, year):
    price_filter = ((buildings.general_type == "Retail")
                    & (buildings.year_built >= year - 5))
    return (buildings
            .non_residential_rent[price_filter]
            .groupby(buildings.zone_id).quantile())


@orca.column('zones_prices', 'office')
def office(buildings, year):
    price_filter = ((buildings.general_type == "Office")
                    & (buildings.year_built >= year - 5))
    return (buildings.non_residential_rent[price_filter]
            .groupby(buildings.zone_id).quantile())


@orca.column('zones_prices', 'industrial')
def industrial(buildings, year):
    price_filter = ((buildings.general_type == "Industrial")
                    & (buildings.year_built >= year - 5))
    return (buildings.non_residential_rent[price_filter]
            .groupby(buildings.zone_id).quantile())


@orca.column('zones_prices', 'zone_id')
def zone_id(zones):
    return zones.index


#####################
# BUILDINGS VARIABLES
#####################


@orca.column('buildings', 'unit_sqft', cache=True, cache_scope='iteration')
def unit_sqft(buildings):
    return buildings.building_sqft / buildings.residential_units.replace(0, 1)


@orca.column('buildings', 'unit_lot_size', cache=True, cache_scope='iteration')
def unit_lot_size(buildings, parcels):
    return (misc.reindex(parcels.parcel_size, buildings.parcel_id)
            / buildings.residential_units.replace(0, 1))


@orca.column('buildings', 'lot_size_per_unit', cache=True)
def lot_size_per_unit(buildings, parcels):
    return misc.reindex(parcels.lot_size_per_unit, buildings.parcel_id)


@orca.column('buildings', 'sqft_per_job', cache=True)
def sqft_per_job(buildings, building_sqft_per_job):
    return buildings.building_type_id.fillna(-1).map(building_sqft_per_job)


#####################
# PARCELS VARIABLES
#####################


@orca.column('parcels', 'max_far', cache=True)
def max_far(parcels, scenario, scenario_inputs):
    return (utils.conditional_upzone(scenario,
                                     scenario_inputs,
                                     "max_far", "far_up")
            .reindex(parcels.index).fillna(0))


@orca.column('parcels', 'max_height', cache=True, cache_scope='iteration')
def max_height(parcels, zoning_baseline):
    return zoning_baseline.max_height.reindex(parcels.index).fillna(0)


@orca.column('parcels', 'total_units', cache=True, cache_scope='iteration')
def total_units(parcels, buildings):
    return (buildings.residential_units
            .groupby(buildings.parcel_id).sum()
            .reindex(parcels.index).fillna(0))


@orca.column('parcels', 'ave_sqft_per_unit_placeholder')
def ave_sqft_per_unit_placeholder(parcels):
    return pd.Series(data=1000, index=parcels.index)


# this just changes the column name for reverse compatibility
@orca.column('parcels', 'ave_unit_size')
def ave_unit_size(parcels):
    return parcels.ave_sqft_per_unit


@orca.column('parcels', 'total_residential_units', cache=False)
def total_residential_units(parcels, buildings):
    return (buildings.residential_units
            .groupby(buildings.parcel_id).sum()
            .reindex(parcels.index).fillna(0))


@orca.column('parcels', 'lot_size_per_unit')
def log_size_per_unit(parcels):
    return parcels.parcel_size / parcels.total_residential_units.replace(0, 1)


@orca.column('parcels', 'land_cost')
def land_cost(parcels, parcel_sales_price_sqft_func):
    # TODO
    # this needs to account for cost for the type of building it is
    return ((parcels.total_sqft
            * parcel_sales_price_sqft_func("residential"))
            .reindex(parcels.index).fillna(0))
