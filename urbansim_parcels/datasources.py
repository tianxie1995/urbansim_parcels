from __future__ import print_function, division, absolute_import

import os
import warnings

import orca
import pandas as pd
from urbansim.utils import misc
import yaml

warnings.filterwarnings('ignore', category=pd.io.pytables.PerformanceWarning)
pd.options.mode.chained_assignment = None


@orca.injectable('year')
def year(iter_var):
    return iter_var


@orca.injectable('settings', cache=True)
def settings():
    with open(os.path.join(misc.configs_dir(), "settings.yaml")) as f:
        settings = yaml.load(f)
        # monkey patch on the settings object since it's pretty global
        # but will also be available as injectable
        orca.settings = settings
        return settings


@orca.injectable('run_number', cache=True)
def run_number():
    return misc.get_run_number()


@orca.injectable('store', cache=True)
def hdfstore(settings):
    return pd.HDFStore(
        os.path.join(misc.data_dir(), settings["store"]),
        mode='r')


@orca.injectable("building_type_map")
def building_type_map(settings):
    return settings["building_type_map"]


@orca.injectable("aggregations")
def aggregations(settings):
    if "aggregation_tables" not in settings or \
            settings["aggregation_tables"] is None:
        return []
    return [orca.get_table(tbl) for tbl in settings["aggregation_tables"]]


@orca.injectable('building_sqft_per_job', cache=True)
def building_sqft_per_job(settings):
    return settings['building_sqft_per_job']


@orca.table('buildings', cache=True)
def buildings(store):
    df = store['buildings']
    return df


@orca.table('household_controls', cache=True)
def household_controls():
    df = pd.read_csv(os.path.join(misc.data_dir(), "household_controls.csv"))
    return df.set_index('year')


@orca.table('employment_controls', cache=True)
def employment_controls():
    df = pd.read_csv(os.path.join(misc.data_dir(), "employment_controls.csv"))
    return df.set_index('year')


@orca.table('jobs', cache=True)
def jobs(store):
    df = store['jobs']
    return df


@orca.table('households', cache=True)
def households(store):
    df = store['households']
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


@orca.table("nodes", cache=True)
def nodes():
    return pd.DataFrame()


# this specifies the relationships between tables
orca.broadcast('nodes', 'buildings', cast_index=True, onto_on='node_id')
orca.broadcast('nodes', 'parcels', cast_index=True, onto_on='node_id')
orca.broadcast('parcels', 'buildings', cast_index=True, onto_on='parcel_id')
orca.broadcast(
    'buildings', 'households', cast_index=True, onto_on='building_id')
orca.broadcast('buildings', 'jobs', cast_index=True, onto_on='building_id')
