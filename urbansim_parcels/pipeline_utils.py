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
    Removes parcels associated with development sites currently in the
    pipeline.

    Parameters
    ----------
    parcels : DataFrameWrapper
        Orca DataFrameWrapper for parcel table. Parcels must be indexed by
        parcel_id

    Returns
    -------
    new_sites : DataFrame
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


def split_evenly(df, area_col, other_split_cols, num_splits):
    """
    Example parcel splitting utility function. Splits parcel into equal
    sizes.

    Parameters
    ----------
    df : DataFrame
        DataFrame of parcels to split
    area_col : str
        Name of column that contains land area to split
    other_split_cols : str or list
        Name(s) of other columns to split (like land cost)
    num_splits : int
        Number of splits to create

    Returns
    -------
    split_df : DataFrame
        DataFrame of sites split from parcel, with a new index and
        parcel_id included as a column

    """
    if not isinstance(other_split_cols, list):
        other_split_cols = [other_split_cols]
    split_cols = [area_col] + other_split_cols

    repeated_index = np.repeat(df.index.values, num_splits)
    split_df = df.loc[repeated_index].copy()
    split_df.index.name = 'parcel_id'
    split_df.reset_index(inplace=True)

    # Divide values in certain columns
    split_df.loc[:, split_cols] /= num_splits
    return split_df


def split_by_size(df, area_col, other_split_cols, size):
    """
    Example parcel splitting utility function. Splits parcel into sites of
    defined sizes. There may be unused area left over from the split.

    Parameters
    ----------
    df : DataFrame
        DataFrame of parcels to split
    area_col : str
        Name of column that contains land area to split
    other_split_cols : str or list
        Name(s) of other columns to split (like land cost)
    size : numeric
        Size of split sites, in units of area_col

    Returns
    -------
    split_df : DataFrame
        DataFrame of sites split from parcel, with a new index and
        parcel_id included as a column
    """
    if not isinstance(other_split_cols, list):
        other_split_cols = [other_split_cols]
    split_cols = [area_col] + other_split_cols

    df['sites_available'] = (df[area_col] // size).astype(int)

    repeated_index = np.repeat(df.index.values, df.sites_available.values)
    split_df = df.loc[repeated_index].copy()
    split_df.index.name = 'parcel_id'
    split_df.reset_index(inplace=True)

    for col in split_cols:
        split_df[col] = split_df[col] / split_df.sites_available
    return split_df


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
    add_buildings = ds[buildings.columns]
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
    pipeline = orca.get_table(pipeline_name).to_frame()
    sites = orca.get_table(sites_name).to_frame()

    buildings = orca.get_table(buildings_name)
    buildings = buildings.to_frame(buildings.local_columns)

    year = orca.get_injectable(year_name)

    results = build_from_pipeline(pipeline, sites, buildings, year)
    new_pipeline, new_sites, new_buildings = results

    orca.add_table(pipeline_name, new_pipeline)
    orca.add_table(sites_name, new_sites)
    orca.add_table(buildings_name, new_buildings)
