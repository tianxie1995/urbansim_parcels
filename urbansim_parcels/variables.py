from __future__ import print_function, division, absolute_import

import random
import numpy as np
import orca
import pandas as pd
from urbansim.utils import misc
from urbansim.utils import networks

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
def parcel_average_occupancy(df, pf):

    for use in pf.uses:
        buildings = orca.get_table('building_occupancy').to_frame(
            ['zone_id', 'parcel_id', 'occupancy_res', 'occupancy_nonres'])

        occ_col = ('occupancy_res' if use == 'residential'
                   else 'occupancy_nonres')

        # Series of average occupancy indexed by zone
        occupancy_by_zone = (buildings[['zone_id', occ_col]]
                             .groupby('zone_id')
                             .agg('mean')
                             [occ_col])

        # Add series above to buildings table
        buildings['zonal_occupancy'] = misc.reindex(occupancy_by_zone,
                                                    buildings.zone_id)

        # Group buildings table to parcels
        parcel_occupancy = (buildings[['zonal_occupancy', 'parcel_id']]
                            .groupby('parcel_id')
                            .agg('mean')
                            .zonal_occupancy)

        occ_var = 'occ_{}'.format(use)
        df[occ_var] = parcel_occupancy

    return df


@orca.injectable('modify_df_occupancy', autocall=False)
def modify_df_occupancy(self, form, df):
    """
    Passed to modify_df parameter of SqftProForma.lookup().

    Requires df to have a set of columns, one for each of the uses passed in
    the configuration, where values are proportion of new development that
    would be expected to be occupied, and names have "occ_" prefix with use.
    Typical names would be "occ_residential", "occ_retail", etc.
    """

    occupancies = ['occ_{}'.format(use) for use in self.uses]
    if set(occupancies).issubset(set(df.columns.tolist())):
        df['weighted_occupancy'] = np.dot(
            df[occupancies],
            self.forms[form])
    else:
        df['weighted_occupancy'] = 1.0

    df = df.loc[df.weighted_occupancy > .50]

    return df


@orca.injectable('modify_revenues_occupancy', autocall=False)
def modify_revenues_occupancy(self, form, df, revenues):
    """
    Passed to modify_revenues parameter of SqftProForma.lookup().
    Note that the weighted_occupancy column must be transformed into values
    because revenues is a numpy ndarray.
    """
    return revenues * df.weighted_occupancy.values


@orca.injectable('res_selection', autocall=False)
def res_selection(self, df, p):
    min_profit_per_sqft = 20
    print("BUILDING ALL BUILDINGS WITH PROFIT > ${:.2f} / sqft"
          .format(min_profit_per_sqft))
    profitable = df.loc[df.max_profit_per_size > min_profit_per_sqft]
    build_idx = profitable.index.values
    return build_idx


@orca.injectable('nonres_selection', autocall=False)
def nonres_selection(self, df, p):
    min_profit_per_sqft = 10
    print("BUILDING ALL BUILDINGS WITH PROFIT > ${:.2f} / sqft"
          .format(min_profit_per_sqft))
    profitable = df.loc[df.max_profit_per_size > min_profit_per_sqft]
    build_idx = profitable.index.values
    return build_idx


@orca.injectable('custom_selection', autocall=False)
def custom_selection(self, df, p):
    profit_cost_ratio = .10
    minimum_profit = 100000
    print("BUILDING ALL BUILDINGS WITH PROFIT TO COST RATIO > {:.0f}%"
          " AND PROFIT > ${:,}"
          .format(profit_cost_ratio * 100, minimum_profit))
    condition = ((df.max_profit / df.total_cost > profit_cost_ratio)
                 & (df.max_profit > minimum_profit))
    profitable = df.loc[condition]
    build_idx = profitable.index.values
    return build_idx


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
