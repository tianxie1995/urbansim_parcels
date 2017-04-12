from __future__ import print_function, division, absolute_import


import orca

from urbansim_parcels import utils
from urbansim_parcels import datasources
from urbansim_parcels import variables
from sf_example import custom_datasources
from sf_example import custom_variables


@orca.step("diagnostic_output")
def diagnostic_output(households, buildings, parcels, zones, year, summary):
    households = households.to_frame()
    buildings = buildings.to_frame()
    parcels = parcels.to_frame()
    zones = zones.to_frame()

    zones['zoned_du'] = parcels.groupby('zone_id').zoned_du.sum()
    zones['zoned_du_underbuild'] = (parcels.groupby('zone_id').
                                    zoned_du_underbuild.sum())
    zones['zoned_du_underbuild_ratio'] = (zones.zoned_du_underbuild /
                                          zones.zoned_du)

    zones['residential_units'] = (buildings.groupby('zone_id').
                                  residential_units.sum())
    zones['non_residential_sqft'] = (buildings.groupby('zone_id').
                                     non_residential_sqft.sum())

    zones['retail_sqft'] = (buildings.query('general_type == "Retail"').
                            groupby('zone_id').non_residential_sqft.sum())
    zones['office_sqft'] = (buildings.query('general_type == "Office"').
                            groupby('zone_id').non_residential_sqft.sum())
    zones['industrial_sqft'] = (buildings.query('general_type == "Industrial"')
                                .groupby('zone_id').non_residential_sqft.sum())

    zones['average_income'] = households.groupby('zone_id').income.quantile()
    zones['household_size'] = households.groupby('zone_id').persons.quantile()

    zones['building_count'] = (buildings.
                               query('general_type == "Residential"').
                               groupby('zone_id').size())
    zones['residential_price'] = (buildings.
                                  query('general_type == "Residential"').
                                  groupby('zone_id').
                                  residential_price.quantile())
    zones['retail_rent'] = (buildings[buildings.general_type == "Retail"].
                            groupby('zone_id').
                            non_residential_price.quantile())
    zones['office_rent'] = (buildings[buildings.general_type == "Office"].
                            groupby('zone_id').
                            non_residential_price.quantile())
    zones['industrial_rent'] = (buildings[
                                    buildings.general_type == "Industrial"
                                ].groupby('zone_id').
                                non_residential_price.quantile())

    summary.add_zone_output(zones, "diagnostic_outputs", year)


@orca.step('regional_occupancy')
def regional_occupancy(year, occupancy, buildings,
                       households, jobs, new_households, new_jobs):
    occupancy = utils.run_occupancy(year, occupancy, buildings,
                                    households, new_households,
                                    jobs, new_jobs,
                                    buildings.sqft_per_job, 20)
    print(occupancy)


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
        parcels.ave_sqft_per_unit_placeholder,
        parcels.total_residential_units,
        'res_developer.yaml',
        year=year,
        form_to_btype_callback=form_to_btype_func,
        add_more_columns_callback=add_extra_columns_func,
        num_units_to_build=None,
        profit_to_prob_func=None)

    summary.add_parcel_output(new_buildings)


@orca.step('residential_developer_profit')
def residential_developer_profit(feasibility, households, buildings,
                                 parcels, year, summary,
                                 form_to_btype_func,
                                 add_extra_columns_func):
    new_buildings = utils.run_developer(
        "residential",
        households,
        buildings,
        'residential_units',
        feasibility,
        parcels.parcel_size,
        parcels.ave_sqft_per_unit_placeholder,
        parcels.total_residential_units,
        'res_developer.yaml',
        year=year,
        form_to_btype_callback=form_to_btype_func,
        add_more_columns_callback=add_extra_columns_func,
        num_units_to_build=None,
        profit_to_prob_func=None,
        min_profit_per_sqft=20.0)

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
        parcels.ave_sqft_per_unit_placeholder,
        parcels.total_job_spaces,
        'nonres_developer.yaml',
        year=year,
        form_to_btype_callback=form_to_btype_func,
        add_more_columns_callback=add_extra_columns_func,
        num_units_to_build=None,
        profit_to_prob_func=None)

    summary.add_parcel_output(new_buildings)


@orca.step('non_residential_developer_profit')
def non_residential_developer_profit(feasibility, jobs, buildings,
                                     parcels, year, summary,
                                     form_to_btype_func,
                                     add_extra_columns_func):
    new_buildings = utils.run_developer(
        ["office", "retail", "industrial"],
        jobs,
        buildings,
        'job_spaces',
        feasibility,
        parcels.parcel_size,
        parcels.ave_sqft_per_unit_placeholder,
        parcels.total_job_spaces,
        'nonres_developer.yaml',
        year=year,
        form_to_btype_callback=form_to_btype_func,
        add_more_columns_callback=add_extra_columns_func,
        num_units_to_build=None,
        profit_to_prob_func=None,
        min_profit_per_sqft=20.0)

    summary.add_parcel_output(new_buildings)

