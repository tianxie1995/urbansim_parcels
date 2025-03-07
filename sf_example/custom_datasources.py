from __future__ import print_function, division, absolute_import

import os
import uuid
import warnings

import numpy as np
import orca
import pandas as pd
from urbansim.utils import misc

from urbansim_parcels import utils
from sf_example import custom_utils

warnings.filterwarnings('ignore', category=pd.io.pytables.PerformanceWarning)
pd.options.mode.chained_assignment = None


@orca.injectable('uuid', cache=True)
def uuid_hex():
    return uuid.uuid4().hex


@orca.injectable("scenario")
def scenario(settings):
    return settings["scenario"]


@orca.injectable("summary", cache=True)
def simulation_summary_data(run_number):
    return custom_utils.SimulationSummaryData(run_number)


@orca.injectable("scenario_inputs")
def scenario_inputs(settings):
    return settings["scenario_inputs"]


@orca.injectable('form_to_btype')
def form_to_btype(settings):
    return settings["form_to_btype"]


@orca.table('buildings', cache=True)
def buildings(store, households, jobs, building_sqft_per_job, settings):
    df = store['buildings']

    if settings.get("set_nan_price_to_zero", False):
        for col in ["residential_price", "non_residential_price"]:
            df[col] = 0

    if settings.get("reconcile_residential_units_and_households", False):
        # prevent overfull buildings (residential)
        df["residential_units"] = pd.concat(
            [df.residential_units, households.building_id.value_counts()],
            axis=1).max(axis=1)

    if settings.get("reconcile_non_residential_sqft_and_jobs", False):
        # prevent overfull buildings (non-residential)
        tmp_df = pd.concat([
            df.non_residential_sqft,
            jobs.building_id.value_counts() *
            df.building_type_id.fillna(-1).map(building_sqft_per_job)
        ], axis=1)
        df["non_residential_sqft"] = tmp_df.max(axis=1).apply(np.ceil)

    fill_nas_cfg = settings.get("table_reprocess", None)
    if fill_nas_cfg is not None:
        fill_nas_cfg = fill_nas_cfg.get("buildings", None)
    if fill_nas_cfg is not None:
        df = utils.table_reprocess(fill_nas_cfg, df)

    return df


@orca.table('jobs', cache=True)
def jobs(store, settings):
    df = store['jobs']

    if settings.get("remove_invalid_building_ids", True):
        # have to do it this way to prevent circular reference
        df.building_id.loc[~df.building_id.isin(store['buildings'].index)] = -1

    fill_nas_cfg = settings.get("table_reprocess", {}).get("jobs", None)
    if fill_nas_cfg is not None:
        df = utils.table_reprocess(fill_nas_cfg, df)

    return df


@orca.table('households', cache=True)
def households(store, settings):
    df = store['households']

    if settings.get("remove_invalid_building_ids", True):
        # have to do it this way to prevent circular reference
        df.building_id.loc[~df.building_id.isin(store['buildings'].index)] = -1

    fill_nas_cfg = settings.get("table_reprocess", None)
    if fill_nas_cfg is not None:
        fill_nas_cfg = fill_nas_cfg.get("households", None)
    if fill_nas_cfg is not None:
        df = utils.table_reprocess(fill_nas_cfg, df)

    return df


@orca.table('parcels', cache=True)
def parcels(store):
    df = store['parcels']
    return df


# these are shapes - "zones" in the bay area
@orca.table('zones', cache=True)
def zones(store):
    df = store['zones']
    return df


# starts with the same underlying shapefile,
# but is used later in the simulation
@orca.table('zones_prices', cache=True)
def zones_prices(store):
    df = store['zones']
    return df


# this is the mapping of parcels to zoning attributes
@orca.table('zoning_for_parcels', cache=True)
def zoning_for_parcels(store):
    df = store['zoning_for_parcels']
    df = df.reset_index().drop_duplicates(subset='parcel').set_index('parcel')
    return df


# this is the actual zoning
@orca.table('zoning', cache=True)
def zoning(store):
    df = store['zoning']
    return df


# zoning for use in the "baseline" scenario
# comes in the hdf5
@orca.table('zoning_baseline', cache=True)
def zoning_baseline(zoning, zoning_for_parcels):
    df = pd.merge(zoning_for_parcels.to_frame(),
                  zoning.to_frame(),
                  left_on='zoning',
                  right_index=True)
    return df


# these are dummy returns that last until accessibility runs
@orca.table("nodes", cache=True)
def nodes():
    return pd.DataFrame()


@orca.table("logsums", cache=True)
def logsums(settings):
    logsums_index = settings.get("logsums_index_col", "taz")
    return pd.read_csv(os.path.join(misc.data_dir(),
                                    'logsums.csv'),
                       index_col=logsums_index)


# this specifies the relationships between tables
orca.broadcast('nodes', 'buildings', cast_index=True, onto_on='node_id')
orca.broadcast('nodes', 'parcels', cast_index=True, onto_on='node_id')
orca.broadcast('zones', 'buildings', cast_index=True, onto_on='zone_id')
orca.broadcast('zones_prices', 'buildings', cast_index=True, onto_on='zone_id')
orca.broadcast('logsums', 'buildings', cast_index=True, onto_on='zone_id')
orca.broadcast('logsums', 'parcels', cast_index=True, onto_on='zone_id')
orca.broadcast('parcels', 'buildings', cast_index=True, onto_on='parcel_id')
orca.broadcast(
    'buildings', 'households', cast_index=True, onto_on='building_id')
orca.broadcast('buildings', 'jobs', cast_index=True, onto_on='building_id')
