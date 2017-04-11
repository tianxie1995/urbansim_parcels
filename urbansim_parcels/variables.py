from __future__ import print_function, division, absolute_import

import random
import numpy as np
import orca
import pandas as pd
from urbansim.utils import misc

from urbansim_parcels import datasources
from urbansim_parcels import utils


#####################
# CALLBACK FUNCTIONS
#####################

@orca.injectable('parcel_sales_price_sqft_func', autocall=False)
def parcel_average_price(use):
    return misc.reindex(orca.get_table('zones_prices')[use],
                        orca.get_table('parcels').zone_id)


@orca.injectable('parcel_is_allowed_func', autocall=False)
def parcel_is_allowed(form):
    form_to_btype = orca.get_injectable("form_to_btype")
    # we have zoning by building type but want
    # to know if specific forms are allowed
    allowed = [orca.get_table('zoning_baseline')
               ['type%d' % typ] == 't' for typ in form_to_btype[form]]
    return pd.concat(allowed, axis=1).max(axis=1).\
        reindex(orca.get_table('parcels').index).fillna(False)


@orca.injectable('form_to_btype_func', autocall=False)
def random_type(row):
    form = row['form']
    form_to_btype = orca.get_injectable("form_to_btype")
    return random.choice(form_to_btype[form])


@orca.injectable('parcel_occupancy_func', autocall=False)
def parcel_average_occupancy(use, oldest_year):

    households = orca.get_table('households')
    jobs = orca.get_table('jobs')
    buildings = (orca.get_table('buildings')
                 .to_frame(['parcel_id', 'residential_units',
                            'non_residential_sqft', 'sqft_per_job',
                            'zone_id', 'year_built']))
    parcels = orca.get_table('parcels').to_frame(['zone_id'])

    buildings = buildings[buildings.year_built >= oldest_year]

    residential = True if use == 'residential' else False
    agents = (households.to_frame(columns=['building_id'])
              if use == 'residential'
              else jobs.to_frame(columns=['building_id']))

    agents_per_building = agents.building_id.value_counts()

    if residential:
        buildings['occupancy'] = (agents_per_building
                                  / buildings.residential_units)
    else:
        job_sqft_per_building = (agents_per_building
                                 * buildings.sqft_per_job)
        buildings['occupancy'] = (job_sqft_per_building
                                  / buildings.non_residential_sqft)

    buildings['occupancy'] = buildings['occupancy'].clip(upper=1.0)

    # Series of average occupancy indexed by zone
    occupancy_by_zone = (buildings[['zone_id', 'occupancy']]
                         .groupby('zone_id')
                         .agg('mean')
                         .occupancy)

    # Add series above to buildings table
    buildings['zonal_occupancy'] = misc.reindex(occupancy_by_zone,
                                                buildings.zone_id)

    # Group buildings table to parcels
    parcel_occupancy = (buildings[['zonal_occupancy', 'parcel_id']]
                        .groupby('parcel_id')
                        .agg('mean')
                        .zonal_occupancy)

    return parcel_occupancy


#####################
# BUILDINGS VARIABLES
#####################


@orca.column('buildings', 'node_id', cache=True)
def node_id(buildings, parcels):
    return misc.reindex(parcels.node_id, buildings.parcel_id)


@orca.column('buildings', 'zone_id', cache=True)
def zone_id(buildings, parcels):
    return misc.reindex(parcels.zone_id, buildings.parcel_id)


@orca.column('buildings', 'general_type', cache=True)
def general_type(buildings, building_type_map):
    return buildings.building_type_id.map(building_type_map)


@orca.column('buildings', 'job_spaces', cache=True)
def job_spaces(buildings):
    return (buildings.non_residential_sqft /
            buildings.sqft_per_job).fillna(0).astype('int')


@orca.column('buildings', 'vacant_residential_units')
def vacant_residential_units(buildings, households):
    return buildings.residential_units.sub(
        households.building_id.value_counts(), fill_value=0)


@orca.column('buildings', 'vacant_job_spaces')
def vacant_job_spaces(buildings, jobs):
    return buildings.job_spaces.sub(
        jobs.building_id.value_counts(), fill_value=0)


#####################
# HOUSEHOLDS VARIABLES
#####################


@orca.column('households', 'income_quartile', cache=True)
def income_quartile(households):
    s = pd.Series(pd.qcut(households.income, 4, labels=False),
                  index=households.index)
    # convert income quartile from 0-3 to 1-4
    s = s.add(1)
    return s


@orca.column('households', 'zone_id', cache=True)
def zone_id(households, buildings):
    return misc.reindex(buildings.zone_id, households.building_id)


@orca.column('households', 'node_id', cache=True)
def node_id(households, buildings):
    return misc.reindex(buildings.node_id, households.building_id)


#####################
# JOBS VARIABLES
#####################


@orca.column('jobs', 'node_id', cache=True)
def node_id(jobs, buildings):
    return misc.reindex(buildings.node_id, jobs.building_id)


@orca.column('jobs', 'zone_id', cache=True)
def zone_id(jobs, buildings):
    return misc.reindex(buildings.zone_id, jobs.building_id)


#####################
# PARCELS VARIABLES
#####################


@orca.column('parcels', 'parcel_size', cache=True)
def parcel_size(parcels, settings):
    return parcels.shape_area * settings.get('parcel_size_factor', 1)


@orca.column('parcels', 'parcel_acres', cache=True)
def parcel_acres(parcels):
    # parcel_size needs to be in sqft
    return parcels.parcel_size / 43560.0


@orca.column('parcels', 'total_job_spaces', cache=False,
             cache_scope='iteration')
def total_job_spaces(parcels, buildings):
    return buildings.job_spaces.groupby(buildings.parcel_id).sum().\
        reindex(parcels.index).fillna(0)


@orca.column('parcels', 'total_sqft', cache=False, cache_scope='iteration')
def total_sqft(parcels, buildings):
    return buildings.building_sqft.groupby(buildings.parcel_id).sum().\
        reindex(parcels.index).fillna(0)


@orca.column('parcels', 'ave_sqft_per_unit')
def ave_sqft_per_unit(parcels, nodes, settings):
    if len(nodes) == 0:
        # if nodes isn't generated yet
        return pd.Series(index=parcels.index)
    s = misc.reindex(nodes.ave_sqft_per_unit, parcels.node_id)
    clip = settings.get("ave_sqft_per_unit_clip", None)
    if clip is not None:
        s = s.clip(lower=clip['lower'], upper=clip['upper'])
    return s


# this just changes the column name for reverse compatibility
@orca.column('parcels', 'ave_unit_size')
def ave_unit_size(parcels):
    return parcels.ave_sqft_per_unit


@orca.column('parcels', 'total_residential_units', cache=False)
def total_residential_units(parcels, buildings):
    return buildings.residential_units.groupby(buildings.parcel_id).sum().\
        reindex(parcels.index).fillna(0)


# returns the oldest building on the land and fills missing values with 9999 -
# for use with historical preservation
@orca.column('parcels', 'oldest_building')
def oldest_building(parcels, buildings):
    return buildings.year_built.groupby(buildings.parcel_id).min().\
        reindex(parcels.index).fillna(9999)
