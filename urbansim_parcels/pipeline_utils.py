from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import orca
import numpy as np
import pandas as pd
from developer import sqftproforma
from developer import develop
from urbansim.utils import misc


def get_new_ids(old_df, new_df, index_name):
    """
    Returns a list of values that can be used as unique indices for new
    entries into a DataFrame.

    Parameters
    ----------
    old_df : DataFrame
        DataFrame to generate new indices for
    new_df : DataFrame
        DataFrame to add new indices to
    index_name : str
        Name for new index on new_df

    Returns
    -------
    new_df : DataFrame
    """
    maxind = np.max(old_df.index.values)
    new_df.reset_index(drop=True, inplace=True)
    new_df.index = new_df.index + maxind + 1
    new_df.index.name = index_name
    return new_df


def add_sites(pipeline, sites, new_sites, project_column=None):
    """
    Adds new sites to existing sites table, creates new projects from new
    sites, and adds new projects to pipeline.

    Parameters
    ----------
    pipeline : DataFrame
        Existing pipeline table
    sites : DataFrame
        Existing sites table
    new_sites : DataFrame
        Sites to add to sites table
    project_column : str, optional
        Name of column in new_sites table. If sites have common values in this
        column, they will be added to the pipeline with the same project_id.

    Returns
    -------
    pipeline : DataFrame
    sites : DataFrame
    """

    if project_column is None:
        new_sites['temp_group'] = pd.Series(range(len(new_sites)))
        project_column = 'temp_group'

    # Get sites ready to add to site table
    new_sites = get_new_ids(sites, new_sites, 'site_id')

    # Get projects ready to add to pipeline table
    new_projects = (new_sites
                    .reset_index(drop=False)    # Keep site_id for counts
                    .groupby(project_column)
                    .agg({'site_id': 'count', 'year_built': 'max'})
                    .reset_index())             # Keep project_column
    new_projects = get_new_ids(pipeline, new_projects, 'project_id')
    new_projects.rename(columns={'site_id': 'sites',
                                 'year_built': 'completion_year'},
                        inplace=True)
    new_projects['sites_active'] = new_projects['sites']
    new_projects['sites_built'] = 0

    # Retroactively match project column to new sites
    project_site_reference = (new_projects[[project_column]]
                              .reset_index()
                              .set_index(project_column))
    new_sites = new_sites.merge(project_site_reference,
                                left_on=project_column, right_index=True)
    new_projects.drop(project_column, axis=1, inplace=True)

    pipeline = pd.concat([pipeline, new_projects], verify_integrity=True)
    sites = pd.concat([sites, new_sites], verify_integrity=True)

    return pipeline, sites


def add_sites_orca(pipeline_name, sites_name, new_sites,
                   project_column=None):
    """
    Wrapper for add_sites function to access Orca tables

    Parameters
    ----------
    pipeline_name : str
        Name of pipeline table in Orca
    sites_name : str
        Name of sites table in Orca
    new_sites : DataFrame
        Sites to add to sites table
    project_column : str, optional
        Name of column in new_sites table. If sites have common values in this
        column, they will be added to the pipeline with the same project_id.
    """
    pipeline = orca.get_table(pipeline_name).to_frame()
    sites = orca.get_table(sites_name).to_frame()
    new_pipeline, new_sites = add_sites(pipeline, sites,
                                        new_sites, project_column)
    orca.add_table(pipeline_name, new_pipeline)
    orca.add_table(sites_name, new_sites)


def remove_pipelined_sites(parcels):
    """
    This has to load the parcel table and remove those already involved
    in pipeline projects

    """

    # Read current dev sites and pipeline
    ds = orca.get_table('dev_sites').to_frame()
    parcels_in_pipeline = ds.parcel_id.unique()

    new_sites = parcels.to_frame().copy()

    print('{} parcels before removing those already in pipeline'
          .format(len(new_sites)))

    # Remove parcels in the pipeline
    new_sites = (new_sites.loc
                 [~new_sites.index.isin(parcels_in_pipeline)])

    print('{} parcels available'.format(len(new_sites)))
    return new_sites


def _create_large_projects(parcels, cfg, parcel_price_callback,
                           parcel_occupancy_callback, start_year,
                           parcel_use_allowed_callback, **kwargs):
    # Read current dev sites and pipeline
    ds = orca.get_table('dev_sites').to_frame()
    sites_in_pipeline = ds.loc[orca.get_injectable('sites_in_pipeline')]
    parcel_ids_in_pipeline = sites_in_pipeline.parcel_id
    candidate_sites = parcels.to_frame().copy()

    print('{} parcels before removing those already in pipeline'
          .format(len(candidate_sites)))

    # Remove parcels in the pipeline
    candidate_sites = (candidate_sites.loc
                       [~candidate_sites.index.isin(parcel_ids_in_pipeline)])

    print('{} parcels before split'
          .format(len(candidate_sites)))

    # TODO Read in user configs

    # TODO Split parcels
    upperbound = 1000000 / 43560
    lowerbound = 200000 / 43560

    large_sites = (candidate_sites.
                   loc[(candidate_sites.parcel_acres > lowerbound)
                       & (candidate_sites.parcel_acres < upperbound)])

    sites = (candidate_sites
             .loc[~candidate_sites.index.isin(large_sites.index.values)])

    print('{} large sites removed'.format(len(large_sites)))
    print('{} sites remaining'.format(len(sites)))

    area_col = 'parcel_acres'
    other_split_cols = ['land_cost']
    splits = 10
    size = 0.25
    # split_sites = _split_evenly(large_sites, area_col,
    #                             other_split_cols, splits)
    split_sites = _split_by_size(large_sites, area_col,
                                 other_split_cols, size)

    print('{} split sites ready for feasibility'.format(len(split_sites)))

    pf = (sqftproforma.SqFtProForma.from_yaml(str_or_buffer=cfg) if cfg
          else sqftproforma.SqFtProForma.from_defaults())
    pf.pass_through = ['parcel_id']
    df = prepare_parcels_for_feasibility(split_sites, parcel_price_callback,
                                         pf, parcel_occupancy_callback,
                                         start_year)

    # lookup_by_form
    lookup_results = {}

    forms = pf.forms_to_test or pf.forms
    for form in forms:
        print("Computing feasibility for form %s" % form)
        allowed = parcel_use_allowed_callback(form).loc[df.parcel_id.unique()]

        newdf = df.loc[misc.reindex(allowed, df.parcel_id)]

        lookup_results[form] = pf.lookup(form, newdf, **kwargs)

    feasibility = pd.concat(lookup_results.values(),
                            keys=lookup_results.keys(),
                            axis=1)

    orca.add_table('split_feasibility', feasibility)

    res = feasibility['residential']
    # build any with combined profit > $500,000
    grouped_profit = res.groupby('parcel_id').agg('sum')
    profitable_parcels = (grouped_profit
                          .loc[grouped_profit.max_profit > 1000000]
                          .index.values)

    # Construction time
    # Just figure out year built for each site

    year = orca.get_injectable('year')
    for parcel in profitable_parcels:
        sites_in_proj = res.loc[res.parcel_id == parcel].index
        # Randomly add between 0 and 3 months to home construction
        add_months = np.random.randint(0, 3, len(sites_in_proj))
        res.loc[sites_in_proj, 'construction_time_mod'] = (
            res.loc[sites_in_proj, 'construction_time'] + add_months)

        res.loc[sites_in_proj, 'year_built'] = (
            res.loc[sites_in_proj, 'construction_time_mod'] // 12 + year)
    # What about existing buildings?

    add_sites_orca('pipeline', 'dev_sites', res, 'parcel_id')

    return sites


def _split_evenly(df, area_col, other_split_cols, num_splits):
    split_cols = [area_col] + other_split_cols

    repeated_index = np.repeat(df.index.values, num_splits)
    split_df = df.loc[repeated_index].copy()
    split_df.index.name = 'parcel_id'
    split_df.reset_index(inplace=True)

    # Divide values in certain columns
    split_df.loc[:, split_cols] /= num_splits
    return split_df


def _split_by_size(df, area_col, other_split_cols, size):
    split_cols = [area_col] + other_split_cols

    df['sites_available'] = (df[area_col] // size).astype(int)

    repeated_index = np.repeat(df.index.values, df.sites_available.values)
    split_df = df.loc[repeated_index].copy()
    split_df.index.name = 'parcel_id'
    split_df.reset_index(inplace=True)

    for col in split_cols:
        split_df[col] = split_df[col] / split_df.sites_available
    return split_df


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


def build_from_pipeline(pipeline, sites, buildings, year):
    """

    Parameters
    ----------
    pipeline : DataFrame
    sites : DataFrame
    buildings : DataFrame
    year : int

    Returns
    -------
    new_pipeline : DataFrame
    new_sites : DataFrame
    new_buildings : DataFrame
    """

    ds = sites.loc[sites.year_built == year]

    # Build sites due to be built this year
    add_buildings = ds.drop('project_id', axis=1)
    add_buildings = get_new_ids(buildings, add_buildings, 'building_id')
    new_buildings = pd.concat([buildings, add_buildings])

    # Mark pipeline with changes
    sites_built_per_project = ds.project_id.value_counts()
    built_projects = sites_built_per_project.index
    pipeline.loc[built_projects, 'sites_built'] += sites_built_per_project
    pipeline.loc[built_projects, 'sites_active'] -= sites_built_per_project

    # Remove from pipeline if finished
    new_pipeline = pipeline.loc[(pipeline.sites_active > 0)
                                & (pipeline.completion_year > year)]

    new_sites = sites.loc[~sites.index.isin(ds.index)]

    return new_pipeline, new_sites, new_buildings


def build_from_pipeline_orca(pipeline_name, sites_name, buildings_name,
                             year_name):
    """
    Wrapper for build_from_pipeline function to access Orca tables

    Parameters
    ----------
    pipeline_name : str
        Name of pipeline table in Orca
    sites_name : str
        Name of sites table in Orca
    buildings_name : str
        Name of buildings table in Orca
    year_name : str
        Name of year injectable in Orca
    """
    table_names = [pipeline_name, sites_name, buildings_name]

    pipeline, sites, buildings = (orca.get_table(name).to_frame()
                                  for name in table_names)
    year = orca.get_injectable(year_name)

    results = build_from_pipeline(pipeline, sites, buildings, year)
    new_pipeline, new_sites, new_buildings = results

    orca.add_table(pipeline_name, new_pipeline)
    orca.add_table(sites_name, new_sites)
    orca.add_table(buildings_name, new_buildings)
