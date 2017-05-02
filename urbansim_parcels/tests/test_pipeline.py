import pytest
import pandas as pd
import orca

from urbansim_parcels import pipeline_utils as pl


@pytest.fixture()
def sites():
    data = {'site_id': [0, 1, 2, 3, 4],
            'parcel_id': [10, 11, 12, 13, 14],
            'project_id': [0, 1, 2, 3, 3],
            'year_built': [2012, 2012, 2013, 2012, 2013]}
    return pd.DataFrame(data).set_index('site_id', drop=True)


@pytest.fixture()
def pipeline():
    data = {'project_id': [0, 1, 2, 3],
            'completion_year': [2012, 2012, 2013, 2013],
            'sites': [1, 1, 1, 2],
            'sites_active': [1, 1, 1, 2],
            'sites_built': [0, 0, 0, 0]}
    return pd.DataFrame(data).set_index('project_id', drop=True)


def check_latest_year(sites, pipeline):
    site_maximums = sites.groupby('project_id').agg('max')
    site_max_years = site_maximums['year_built']
    s1 = site_max_years.sort_index()
    s2 = pipeline.completion_year.sort_index()
    assert s1.equals(s2)


def check_site_count(sites, pipeline):
    site_project_counts = sites.project_id.value_counts()
    s1 = site_project_counts.sort_index()
    s2 = pipeline.sites_active.sort_index()
    assert s1.equals(s2)


def test_initial_tables(sites, pipeline):
    check_latest_year(sites, pipeline)
    check_site_count(sites, pipeline)


@pytest.fixture()
def new_sites():
    data = {'parcel_id': [10, 11, 12, 13, 13],
            'year_built': [2012, 2012, 2012, 2013, 2013]}
    return pd.DataFrame(data)


def test_get_new_index(sites, new_sites):
    df = pl.get_new_ids(sites, new_sites, 'site_id')
    assert df.index.name == 'site_id'
    assert df.index.values.tolist() == list(range(5, 10))


def test_add_sites_single(pipeline, sites, new_sites):
    new_pipeline, new_sites = (
        # Leaving out project_column argument
        # treats new sites as single projects
        pl.add_sites(pipeline, sites, new_sites))

    assert new_pipeline.index.values.tolist() == list(range(9))
    assert new_sites.index.values.tolist() == list(range(10))

    assert new_sites.loc[5, 'project_id'] == 4
    assert new_sites.loc[8, 'project_id'] == 7
    assert new_sites.loc[9, 'project_id'] == 8

    check_latest_year(new_sites, new_pipeline)
    check_site_count(new_sites, new_pipeline)


def test_add_sites_grouped(pipeline, sites, new_sites):

    new_pipeline, new_sites = (
        # Adding "parcel_id" as project_column
        # groups sites into projects
        pl.add_sites(pipeline, sites, new_sites, 'parcel_id'))

    assert new_pipeline.index.values.tolist() == list(range(8))
    assert new_sites.index.values.tolist() == list(range(10))

    assert new_sites.loc[5, 'project_id'] == 4
    assert new_sites.loc[8, 'project_id'] == 7
    assert new_sites.loc[9, 'project_id'] == 7

    check_latest_year(new_sites, new_pipeline)
    check_site_count(new_sites, new_pipeline)


def test_add_sites_orca(pipeline, sites, new_sites):

    orca.add_table('pipeline', pipeline)
    orca.add_table('dev_sites', sites)

    pl.add_sites_orca('pipeline', 'dev_sites', new_sites)

    new_pipeline = orca.get_table('pipeline').to_frame()
    new_sites = orca.get_table('dev_sites').to_frame()

    assert new_pipeline.index.values.tolist() == list(range(9))
    assert new_sites.index.values.tolist() == list(range(10))

    assert new_sites.loc[5, 'project_id'] == 4
    assert new_sites.loc[8, 'project_id'] == 7
    assert new_sites.loc[9, 'project_id'] == 8

    check_latest_year(new_sites, new_pipeline)
    check_site_count(new_sites, new_pipeline)


@pytest.fixture()
def buildings():
    data = {'parcel_id': [20, 21, 22],
            'building_id': [60, 61, 62],
            'year_built': [2012, 2012, 2012]}
    return pd.DataFrame(data).set_index('building_id', drop=True)


def test_build_from_pipeline(pipeline, sites, buildings):

    results = pl.build_from_pipeline(pipeline, sites, buildings, 2012)

    new_pipeline, new_sites, new_buildings = results

    assert new_buildings.index.values.tolist() == list(range(60, 66))
    assert new_buildings.year_built.unique().max() <= 2012
    assert new_sites.index.values.tolist() == [2, 4]
    assert new_pipeline.index.values.tolist() == [2, 3]
    assert new_pipeline.loc[2, 'sites_active'] == 1
    assert new_pipeline.loc[2, 'sites'] == 1
    assert new_pipeline.loc[2, 'sites_built'] == 0
    assert new_pipeline.loc[3, 'sites_active'] == 1
    assert new_pipeline.loc[3, 'sites'] == 2
    assert new_pipeline.loc[3, 'sites_built'] == 1

    check_latest_year(new_sites, new_pipeline)
    check_site_count(new_sites, new_pipeline)


def test_build_from_pipeline_empty(pipeline):

    sites = pd.DataFrame()
    buildings = pd.DataFrame()
    results = pl.build_from_pipeline(pipeline, sites, buildings, 2012)

    assert results is None


def test_build_from_pipeline_orca(pipeline, sites, buildings):

    orca.add_table('pipeline', pipeline)
    orca.add_table('sites', sites)
    orca.add_table('buildings', buildings)
    orca.add_injectable('year', 2012)

    pl.build_from_pipeline_orca('pipeline', 'sites', 'buildings', 'year')

    new_pipeline = orca.get_table('pipeline').to_frame()
    new_sites = orca.get_table('sites').to_frame()
    new_buildings = orca.get_table('buildings').to_frame()

    # Same assertions as above
    assert new_buildings.index.values.tolist() == list(range(60, 66))
    assert new_buildings.year_built.unique().max() <= 2012
    assert new_sites.index.values.tolist() == [2, 4]
    assert new_pipeline.index.values.tolist() == [2, 3]
    assert new_pipeline.loc[2, 'sites_active'] == 1
    assert new_pipeline.loc[2, 'sites'] == 1
    assert new_pipeline.loc[2, 'sites_built'] == 0
    assert new_pipeline.loc[3, 'sites_active'] == 1
    assert new_pipeline.loc[3, 'sites'] == 2
    assert new_pipeline.loc[3, 'sites_built'] == 1

    check_latest_year(new_sites, new_pipeline)
    check_site_count(new_sites, new_pipeline)


def test_build_from_pipeline_orca_empty(pipeline):

    sites = pd.DataFrame()
    buildings = pd.DataFrame()

    orca.add_table('pipeline', pipeline)
    orca.add_table('sites', sites)
    orca.add_table('buildings', buildings)
    orca.add_injectable('year', 2012)

    pl.build_from_pipeline_orca('pipeline', 'sites', 'buildings', 'year')

    new_pipeline = orca.get_table('pipeline').to_frame()
    assert new_pipeline.equals(pipeline)
