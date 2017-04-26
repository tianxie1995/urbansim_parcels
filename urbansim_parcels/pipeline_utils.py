from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import orca
import numpy as np
import pandas as pd
from urbansim.models import RegressionModel
from urbansim.models import SegmentedRegressionModel
from urbansim.models import MNLDiscreteChoiceModel
from urbansim.models import SegmentedMNLDiscreteChoiceModel
from urbansim.models import GrowthRateTransition
from urbansim.models import transition
from urbansim.models.supplydemand import supply_and_demand
from developer import sqftproforma
from developer import develop
from urbansim.utils import misc
from . import utils


def run_feasibility(parcels, parcel_price_callback,
                    parcel_use_allowed_callback,
                    parcel_occupancy_callback=None, start_year=None,
                    cfg=None, **kwargs):
    """
    Execute development feasibility on all parcels

    Parameters
    ----------
    parcels : DataFrame Wrapper
        The data frame wrapper for the parcel data
    parcel_price_callback : function
        A callback which takes each use of the pro forma and returns a series
        with index as parcel_id and value as yearly_rent
    parcel_use_allowed_callback : function
        A callback which takes each form of the pro forma and returns a series
        with index as parcel_id and value and boolean whether the form
        is allowed on the parcel
    parcel_occupancy_callback : function
        A callback which takes each use of the pro forma, along with a start
        year, and returns series with index as parcel_id and value as
        expected occupancy
    start_year : int
        Year to start tracking occupancy
    cfg : str
        The name of the yaml file to read pro forma configurations from

    Returns
    -------
    Adds a table called feasibility to the sim object (returns nothing)
    """

    cfg = misc.config(cfg)
    pf = (sqftproforma.SqFtProForma.from_yaml(str_or_buffer=cfg) if cfg
          else sqftproforma.SqFtProForma.from_defaults())
    sites = _create_development_projects(parcels)
    df = prepare_parcels_for_feasibility(sites, parcel_price_callback,
                                         pf, parcel_occupancy_callback,
                                         start_year)
    feasibility = utils.lookup_by_form(df, parcel_use_allowed_callback,
                                       pf, **kwargs)
    orca.add_table('feasibility', feasibility)


def _create_development_projects(parcels):
    ds = orca.get_table('dev_sites').to_frame()
    sites_in_pipeline = ds.loc[orca.get_injectable('sites_in_pipeline')]
    parcel_ids_in_pipeline = sites_in_pipeline.parcel_id
    candidate_sites = parcels.to_frame().copy()

    # Remove parcels in the pipeline
    candidate_sites = (candidate_sites
                       [~candidate_sites
                        .index.isin(parcel_ids_in_pipeline)])

    # TODO Read in user configs

    # TODO Split parcels

    return candidate_sites


def prepare_parcels_for_feasibility(sites, parcel_price_callback,
                                    pf, parcel_occupancy_callback=None,
                                    start_year=None):
    """
    Prepare parcel DataFrame for feasibility analysis

    Parameters
    ----------
    sites : DataFrame
        DataFrame of development sites
    parcel_price_callback : func
        A callback which takes each use of the pro forma and returns a series
        with index as parcel_id and value as yearly_rent
    pf: SqFtProForma object
        Pro forma object with relevant configurations
    parcel_occupancy_callback : func
        A callback which takes each use of the pro forma, along with a start
        year, and returns series with index as parcel_id and value as
        expected occupancy
    start_year : int
        Year to begin tracking occupancy

    Returns
    -------
    DataFrame of parcels
    """

    if pf.parcel_filter:
        sites = sites.query(pf.parcel_filter)

    current_year = orca.get_injectable('year')

    for use in pf.uses:
        # Add prices
        sites[use] = parcel_price_callback(use)

        # Add occupancies
        if start_year and current_year >= start_year:
            occ_col = 'occ_{}'.format(use)
            sites[occ_col] = parcel_occupancy_callback(use)

    # convert from cost to yearly rent
    if pf.residential_to_yearly and 'residential' in sites.columns:
        sites["residential"] *= pf.cap_rate

    return sites


def merge_dfs(old_df, new_df, index_name):
    """

    Parameters
    ----------
    old_df
    new_df
    index_name

    Returns
    -------
    concat_df : DataFrame
    new_df.index : Index
    """
    maxind = np.max(old_df.index.values)
    new_df = new_df.reset_index(drop=True)
    new_df.index = new_df.index + maxind + 1
    concat_df = pd.concat([old_df, new_df], verify_integrity=True)
    concat_df.index.name = index_name
    return concat_df, new_df.index


def run_developer(forms, agents, buildings, supply_fname, feasibility,
                  parcel_size, ave_unit_size, current_units, cfg,
                  year=None,
                  target_vacancy=0.1, form_to_btype_callback=None,
                  add_more_columns_callback=None,
                  remove_developed_buildings=True,
                  unplace_agents=['households', 'jobs'],
                  num_units_to_build=None, profit_to_prob_func=None,
                  custom_selection_func=None):

    cfg = misc.config(cfg)

    target_units = (
        num_units_to_build or
        utils.compute_units_to_build(len(agents),
                                     buildings[supply_fname].sum(),
                                     target_vacancy))

    dev = develop.Developer.from_yaml(feasibility.to_frame(), forms,
                                      target_units, parcel_size,
                                      ave_unit_size, current_units,
                                      year, str_or_buffer=cfg)

    print("{:,} feasible buildings before running developer".format(
        len(dev.feasibility)))

    new_buildings = dev.pick(profit_to_prob_func, custom_selection_func)
    orca.add_table('new_buildings', new_buildings)
    orca.add_table("feasibility", dev.feasibility)

    if new_buildings is None:
        return

    if len(new_buildings) == 0:
        return new_buildings

    new_sites = (
        process_new_buildings(dev.feasibility, buildings, new_buildings,
                              form_to_btype_callback,
                              add_more_columns_callback,
                              supply_fname, remove_developed_buildings,
                              unplace_agents))

    utils.add_new_units(dev, new_sites)

    # New stuff below

    years_to_build = new_sites.construction_time // 12
    current_year = orca.get_injectable('year')
    new_sites['year_built'] = years_to_build + current_year

    ds = orca.get_table('dev_sites').to_frame()
    all_sites, new_site_ids = merge_dfs(ds, new_sites, 'dev_site_id')

    dp = orca.get_table('dev_projects').to_frame()
    new_projects = pd.DataFrame(index=range(len(new_sites)))
    all_projects, new_project_ids = merge_dfs(dp, new_projects,
                                              'dev_project_id')
    all_sites.loc[new_site_ids, 'dev_project_id'] = new_project_ids

    pl = orca.get_table('pipeline').to_frame()

    new_pipeline = pd.DataFrame(index=range(len(new_projects)))
    all_pipeline, new_pipeline_ids = merge_dfs(pl, new_pipeline, 'pipeline_id')
    all_projects.loc[new_project_ids, 'pipeline_id'] = new_pipeline_ids
    all_sites.loc[new_site_ids, 'pipeline_id'] = new_pipeline_ids

    # add pipeline attributes
    all_pipeline.loc[new_pipeline_ids, 'completion_year'] = (
        all_sites.loc[new_site_ids, ['pipeline_id', 'year_built']]
        .set_index('pipeline_id', drop=True)
        ['year_built'])
    all_pipeline.loc[new_pipeline_ids, 'sites'] = 1

    # drop columns not in original dev sites
    # all_sites = all_sites[ds.columns.tolist()]

    orca.add_table('dev_sites', all_sites)
    orca.add_table('dev_projects', all_projects)
    orca.add_table('pipeline', all_pipeline)

    return all_sites


def process_new_buildings(feasibility, buildings, new_buildings,
                          form_to_btype_callback,
                          add_more_columns_callback,
                          supply_fname, remove_developed_buildings,
                          unplace_agents):
    if form_to_btype_callback is not None:
        new_buildings["building_type_id"] = new_buildings.apply(
            form_to_btype_callback, axis=1)

    if add_more_columns_callback is not None:
        new_buildings = add_more_columns_callback(new_buildings)

    print("Adding {:,} buildings with {:,} {}".format(
        len(new_buildings),
        int(new_buildings[supply_fname].sum()),
        supply_fname))

    print("{:,} feasible buildings after running developer".format(
        len(feasibility)))

    if remove_developed_buildings:
        old_buildings = buildings.to_frame(buildings.local_columns)
        old_buildings = utils._remove_developed_buildings(
            old_buildings, new_buildings, unplace_agents)
        orca.add_table('buildings', old_buildings)

    return new_buildings


def build_from_pipeline(pipeline, dev_projects, dev_sites, year):
    pl, dp, ds = (table.to_frame() for table in
                  [pipeline, dev_projects, dev_sites])

    print('{} projects in pipeline. By completion year:'
          .format(len(pl)))
    print(pipeline.completion_year.value_counts(ascending=True))

    pl_this_year = pl.loc[pl.completion_year == year]
    ds_this_year = ds.loc[(ds.pipeline_id.isin(pl_this_year.index))
                          & (ds.year_built == year)]
    dp_this_year = dp.loc[ds_this_year.dev_project_id.values]

    print('Constructing {} buildings from {} projects'
          .format(len(ds_this_year), len(dp_this_year)))

    old_buildings = orca.get_table('buildings').to_frame()
    all_buildings = utils.merge_buildings(old_buildings, ds_this_year)
    orca.add_table('buildings', all_buildings)

    # Remove built dev sites from table
    new_ds = ds[~ds.index.isin(ds_this_year.index)]
    orca.add_table('dev_sites', new_ds)

    # Remove built dev projects from table
    active_dp = new_ds.dev_project_id.value_counts().index
    new_dp = dp.loc[active_dp]
    orca.add_table('dev_projects', new_dp)

    # Remove built projects from pipeline
    active_pipeline = new_ds.pipeline_id.value_counts().index
    new_pipeline = pl.loc[active_pipeline]
    orca.add_table('pipeline', new_pipeline)

    print('{} projects left in pipeline'.format(len(new_pipeline)))

    return all_buildings
