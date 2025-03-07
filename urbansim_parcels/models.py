from __future__ import print_function, division, absolute_import

import os

import orca
import pandana as pdna
import pandas as pd
import numpy as np
from urbansim.utils import misc
from urbansim.utils import networks

from . import utils
from . import datasources
from . import variables
from . import pipeline_utils as pl


@orca.step('rsh_estimate')
def rsh_estimate(buildings, aggregations):
    return utils.hedonic_estimate("rsh.yaml", buildings, aggregations)


@orca.step('rsh_simulate')
def rsh_simulate(buildings, aggregations):
    return utils.hedonic_simulate("rsh.yaml", buildings, aggregations,
                                  "residential_sales_price")


@orca.step('nrh_estimate')
def nrh_estimate(buildings, aggregations):
    return utils.hedonic_estimate("nrh.yaml", buildings, aggregations)


@orca.step('nrh_simulate')
def nrh_simulate(buildings, aggregations):
    return utils.hedonic_simulate("nrh.yaml", buildings, aggregations,
                                  "non_residential_rent")


@orca.step('hlcm_estimate')
def hlcm_estimate(households, buildings, aggregations):
    return utils.lcm_estimate("hlcm.yaml", households, "building_id",
                              buildings, aggregations)


@orca.step('hlcm_simulate')
def hlcm_simulate(households, buildings, aggregations, settings):
    return utils.lcm_simulate("hlcm.yaml", households, buildings,
                              aggregations,
                              "building_id", "residential_units",
                              "vacant_residential_units",
                              settings.get("enable_supply_correction", None))


@orca.step('elcm_estimate')
def elcm_estimate(jobs, buildings, aggregations):
    return utils.lcm_estimate("elcm.yaml", jobs, "building_id",
                              buildings, aggregations)


@orca.step('elcm_simulate')
def elcm_simulate(jobs, buildings, aggregations):
    return utils.lcm_simulate("elcm.yaml", jobs, buildings, aggregations,
                              "building_id", "job_spaces",
                              "vacant_job_spaces")


@orca.step('households_relocation')
def households_relocation(households, settings):
    rate = settings['rates']['households_relocation']
    return utils.simple_relocation(households, rate, "building_id")


@orca.step('jobs_relocation')
def jobs_relocation(jobs, settings):
    rate = settings['rates']['jobs_relocation']
    return utils.simple_relocation(jobs, rate, "building_id")


@orca.step('households_transition')
def households_transition(households, household_controls, year, settings):
    return utils.full_transition(households,
                                 household_controls,
                                 year,
                                 settings['households_transition'],
                                 "building_id")


@orca.step('simple_households_transition')
def simple_households_transition(households, settings):
    rate = settings['rates']['simple_households_transition']
    return utils.simple_transition(households, rate, "building_id")


@orca.step('jobs_transition')
def jobs_transition(jobs, employment_controls, year, settings):
    return utils.full_transition(jobs,
                                 employment_controls,
                                 year,
                                 settings['jobs_transition'],
                                 "building_id")


@orca.step('simple_jobs_transition')
def simple_jobs_transition(jobs, settings):
    rate = settings['rates']['simple_jobs_transition']
    return utils.simple_transition(jobs, rate, "building_id")


@orca.injectable('net', cache=True)
def build_networks(settings):
    name = settings['build_networks']['name']
    st = pd.HDFStore(os.path.join(misc.data_dir(), name), "r")
    nodes, edges = st.nodes, st.edges
    net = pdna.Network(nodes["x"], nodes["y"], edges["from"], edges["to"],
                       edges[["weight"]])
    net.precompute(settings['build_networks']['max_distance'])
    return net


@orca.step('neighborhood_vars')
def neighborhood_vars(net):
    nodes = networks.from_yaml(net, "neighborhood_vars.yaml")
    nodes = nodes.fillna(0)
    print(nodes.describe())
    orca.add_table("nodes", nodes)


@orca.step('price_vars')
def price_vars(net):
    nodes2 = networks.from_yaml(net, "price_vars.yaml")
    nodes2 = nodes2.fillna(0)
    print(nodes2.describe())
    nodes = orca.get_table('nodes')
    nodes = nodes.to_frame().join(nodes2)
    orca.add_table("nodes", nodes)


@orca.step('occupancy_vars')
def occupancy_vars(year):
    oldest_year = year - 20

    building_occupancy = utils.building_occupancy(oldest_year)
    orca.add_table('building_occupancy', building_occupancy)

    res_mean = building_occupancy.occupancy_res.mean()
    print('Average residential occupancy in {} for buildings built'
          ' since {}: {:.2f}%'.format(year, oldest_year, res_mean * 100))

    nonres_mean = building_occupancy.occupancy_nonres.mean()
    print('Average non-residential occupancy in {} for buildings built'
          ' since {}: {:.2f}%'.format(year, oldest_year, nonres_mean * 100))


@orca.step('occupancy_vars_network')
def occupancy_vars_network(year, net):
    """
    Alternative step that additionally aggregates occupancy along the Pandana
    network (the basic occupancy_vars step above relies on additional
    aggregation in the parcel_average_occupancy callback function).
    """

    oldest_year = year - 20
    building_occupancy = utils.building_occupancy(oldest_year)
    orca.add_table('building_occupancy', building_occupancy)

    res_mean = building_occupancy.occupancy_res.mean()
    print('Average residential occupancy in {} for buildings built'
          ' since {}: {:.2f}%'.format(year, oldest_year, res_mean * 100))

    nonres_mean = building_occupancy.occupancy_nonres.mean()
    print('Average non-residential occupancy in {} for buildings built'
          ' since {}: {:.2f}%'.format(year, oldest_year, nonres_mean * 100))

    nodes2 = networks.from_yaml(net, "occupancy_vars.yaml")
    nodes2 = nodes2.fillna(0)
    print(nodes2.describe())
    nodes = orca.get_table('nodes')
    nodes = nodes.to_frame().join(nodes2)
    orca.add_table("nodes", nodes)


@orca.step('feasibility')
def feasibility(parcels,
                parcel_sales_price_sqft_func,
                parcel_is_allowed_func):
    utils.run_feasibility(parcels,
                          parcel_sales_price_sqft_func,
                          parcel_is_allowed_func,
                          cfg='proforma.yaml')


@orca.step('feasibility_with_pipeline')
def feasibility_with_pipeline(parcels,
                              parcel_sales_price_sqft_func,
                              parcel_is_allowed_func):
    utils.run_feasibility(parcels,
                          parcel_sales_price_sqft_func,
                          parcel_is_allowed_func,
                          pipeline=True,
                          cfg='proforma.yaml')


@orca.step('feasibility_large_parcels')
def feasibility_large_parcels(parcels,
                              parcel_sales_price_sqft_func,
                              parcel_is_allowed_func,
                              large_parcel_split_func):
    utils.run_feasibility(parcels,
                          parcel_sales_price_sqft_func,
                          parcel_is_allowed_func,
                          cfg='proforma_split.yaml',
                          parcel_id_col='parcel_id',
                          parcel_custom_callback=large_parcel_split_func)


@orca.step('feasibility_with_occupancy')
def feasibility_with_occupancy(parcels,
                               parcel_sales_price_sqft_func,
                               parcel_is_allowed_func,
                               parcel_occupancy_func,
                               modify_df_occupancy,
                               modify_revenues_occupancy):
    utils.run_feasibility(parcels,
                          parcel_sales_price_sqft_func,
                          parcel_is_allowed_func,
                          cfg='proforma.yaml',
                          modify_df=modify_df_occupancy,
                          modify_revenues=modify_revenues_occupancy,
                          parcel_custom_callback=parcel_occupancy_func)


@orca.injectable("add_extra_columns_func", autocall=False)
def add_extra_columns(df):
    for col in ["residential_price", "non_residential_price",
                'residential_sales_price', 'non_residential_rent']:
        df[col] = 0
    return df


@orca.step('developer_large_parcels')
def developer_large_parcels(feasibility, households, jobs, buildings, parcels,
                            year, summary, form_to_btype_func,
                            add_extra_columns_func,
                            large_parcel_selection_func):
    new_res = utils.run_developer(
        "residential",
        households,
        buildings,
        'residential_units',
        feasibility,
        parcels.parcel_size,
        parcels.ave_sqft_per_unit,
        parcels.total_residential_units,
        'res_developer.yaml',
        year=year,
        form_to_btype_callback=form_to_btype_func,
        add_more_columns_callback=add_extra_columns_func,
        custom_selection_func=large_parcel_selection_func,
        pipeline=True)

    summary.add_parcel_output(new_res)

    new_nonres = utils.run_developer(
        ["office", "retail", "industrial"],
        jobs,
        buildings,
        'job_spaces',
        feasibility,
        parcels.parcel_size,
        parcels.ave_sqft_per_unit,
        parcels.total_job_spaces,
        'nonres_developer.yaml',
        year=year,
        target_vacancy=.21,
        form_to_btype_callback=form_to_btype_func,
        add_more_columns_callback=add_extra_columns_func,
        custom_selection_func=large_parcel_selection_func,
        pipeline=True)

    summary.add_parcel_output(new_nonres)


@orca.step('residential_developer')
def residential_developer(feasibility, households, buildings, parcels, year,
                          summary, form_to_btype_func, add_extra_columns_func):
    new_buildings = utils.run_developer(
        "residential",
        households,
        buildings,
        'residential_units',
        feasibility,
        parcels.parcel_size,
        parcels.ave_sqft_per_unit,
        parcels.total_residential_units,
        'res_developer.yaml',
        year=year,
        form_to_btype_callback=form_to_btype_func,
        add_more_columns_callback=add_extra_columns_func)

    summary.add_parcel_output(new_buildings)


@orca.step('residential_developer_pipeline')
def residential_developer_pipeline(feasibility, households, buildings, parcels,
                                   year, summary, form_to_btype_func,
                                   add_extra_columns_func):
    new_buildings = utils.run_developer(
        "residential",
        households,
        buildings,
        'residential_units',
        feasibility,
        parcels.parcel_size,
        parcels.ave_sqft_per_unit,
        parcels.total_residential_units,
        'res_developer.yaml',
        year=year,
        form_to_btype_callback=form_to_btype_func,
        add_more_columns_callback=add_extra_columns_func,
        pipeline=True)

    summary.add_parcel_output(new_buildings)


@orca.step('residential_developer_profit')
def residential_developer_profit(feasibility, households, buildings,
                                 parcels, year, summary,
                                 form_to_btype_func, add_extra_columns_func,
                                 res_selection):
    new_buildings = utils.run_developer(
        "residential",
        households,
        buildings,
        'residential_units',
        feasibility,
        parcels.parcel_size,
        parcels.ave_sqft_per_unit,
        parcels.total_residential_units,
        'res_developer.yaml',
        year=year,
        form_to_btype_callback=form_to_btype_func,
        add_more_columns_callback=add_extra_columns_func,
        custom_selection_func=res_selection)

    summary.add_parcel_output(new_buildings)


@orca.step('non_residential_developer')
def non_residential_developer(feasibility, jobs, buildings, parcels, year,
                              summary, form_to_btype_func,
                              add_extra_columns_func):
    new_buildings = utils.run_developer(
        ["office", "retail", "industrial"],
        jobs,
        buildings,
        'job_spaces',
        feasibility,
        parcels.parcel_size,
        parcels.ave_sqft_per_unit,
        parcels.total_job_spaces,
        'nonres_developer.yaml',
        year=year,
        form_to_btype_callback=form_to_btype_func,
        add_more_columns_callback=add_extra_columns_func)

    summary.add_parcel_output(new_buildings)


@orca.step('non_residential_developer_pipeline')
def non_residential_developer_pipeline(feasibility, jobs, buildings, parcels,
                                       year, summary, form_to_btype_func,
                                       add_extra_columns_func):
    new_buildings = utils.run_developer(
        ["office", "retail", "industrial"],
        jobs,
        buildings,
        'job_spaces',
        feasibility,
        parcels.parcel_size,
        parcels.ave_sqft_per_unit,
        parcels.total_job_spaces,
        'nonres_developer.yaml',
        year=year,
        target_vacancy=.21,
        form_to_btype_callback=form_to_btype_func,
        add_more_columns_callback=add_extra_columns_func,
        pipeline=True)

    summary.add_parcel_output(new_buildings)


@orca.step('non_residential_developer_profit')
def non_residential_developer_profit(feasibility, jobs, buildings,
                                     parcels, year, summary,
                                     form_to_btype_func,
                                     add_extra_columns_func, nonres_selection):
    new_buildings = utils.run_developer(
        ["office", "retail", "industrial"],
        jobs,
        buildings,
        'job_spaces',
        feasibility,
        parcels.parcel_size,
        parcels.ave_sqft_per_unit,
        parcels.total_job_spaces,
        'nonres_developer.yaml',
        year=year,
        form_to_btype_callback=form_to_btype_func,
        add_more_columns_callback=add_extra_columns_func,
        custom_selection_func=nonres_selection)

    summary.add_parcel_output(new_buildings)


@orca.step('build_from_pipeline')
def build_from_pipeline():
    pl.build_from_pipeline_orca('pipeline', 'dev_sites', 'buildings', 'year')


@orca.step("scheduled_development_events")
def scheduled_development_events(buildings, development_projects, summary,
                                 year):
    dps = development_projects.to_frame().query("year_built == %d" % year)

    if len(dps) == 0:
        return

    new_buildings = utils.scheduled_development_events(
        buildings, dps,
        remove_developed_buildings=True,
        unplace_agents=[
            'households',
            'jobs'])

    summary.add_parcel_output(new_buildings)
