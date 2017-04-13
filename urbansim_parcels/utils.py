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


def conditional_upzone(scenario, scenario_inputs, attr_name, upzone_name):
    """

    Parameters
    ----------
    scenario : str
        The name of the active scenario (set to "baseline" if no scenario
        zoning)
    scenario_inputs : dict
        Dictionary of scenario options - keys are scenario names and values
        are also dictionaries of key-value paris for scenario inputs.  Right
        now "zoning_table_name" should be set to the table that contains the
        scenario based zoning for that scenario
    attr_name : str
        The name of the attribute in the baseline zoning table
    upzone_name : str
        The name of the attribute in the scenario zoning table

    Returns
    -------
    The new zoning per parcel which is increased if the scenario based
    zoning is higher than the baseline zoning
    """
    zoning_baseline = orca.get_table(
        scenario_inputs["baseline"]["zoning_table_name"])
    attr = zoning_baseline[attr_name]
    if scenario != "baseline":
        zoning_scenario = orca.get_table(
            scenario_inputs[scenario]["zoning_table_name"])
        upzone = zoning_scenario[upzone_name].dropna()
        # need to leave nas as nas - if the density is unrestricted before
        # it should be unrestricted now - so nas in the first series need
        # to be left, but nas in the second series need to be ignored
        # there might be a better way to express this
        attr = pd.concat(
            [attr, upzone.fillna(attr)], axis=1).max(skipna=True, axis=1)
    return attr


def enable_logging():
    """
    A quick shortcut to enable logging at log level INFO
    """
    from urbansim.utils import logutil
    logutil.set_log_level(logutil.logging.INFO)
    logutil.log_to_stream()


def check_nas(df):
    """
    Checks for nas and errors if they are found (also prints a report on how
    many nas are found in each column)

    Parameters
    ----------
    df : DataFrame
        DataFrame to check for nas

    Returns
    -------
    Nothing
    """
    df_cnt = len(df)
    fail = False

    df = df.replace([np.inf, -np.inf], np.nan)
    for col in df.columns:
        s_cnt = df[col].count()
        if df_cnt != s_cnt:
            fail = True
            print("Found {:d} nas or inf (out of {:d}) in column {:s}".format(
                df_cnt - s_cnt, df_cnt, col))

    assert not fail, "NAs were found in dataframe, please fix"


def table_reprocess(cfg, df):
    """
    Reprocesses a table with the given configuration, mainly by filling nas
    with the given configuration.

    Parameters
    ----------
    cfg : dict
        The configuration is specified as a nested dictionary, javascript
        style, and a simple config is given below.  Most parameters should be
        fairly self-explanatory.  "filter" filters the dataframe using the
        query command in Pandas.  The "fill_nas" parameter is another
        dictionary which takes each column and specifies how to fill nas -
        options include "drop", "zero", "median", "mode", and "mean".  The
        "type" must also be specified since items like "median" usually
        return floats but the result is often desired to be an "int" - the
        type is thus specified to avoid ambiguity.::

            {
                "filter": "building_type_id >= 1 and building_type_id <= 14",
                "fill_nas": {
                    "building_type_id": {
                        "how": "drop",
                        "type": "int"
                    },
                    "residential_units": {
                        "how": "zero",
                        "type": "int"
                    },
                    "year_built": {
                        "how": "median",
                        "type": "int"
                    },
                    "building_type_id": {
                        "how": "mode",
                        "type": "int"
                    }
                }
            }

    df : DataFrame to process

    Returns
    -------
    New DataFrame which is reprocessed according the configuration
    """
    df_cnt = len(df)

    if "filter" in cfg:
        df = df.query(cfg["filter"])

    assert "fill_nas" in cfg
    cfg = cfg["fill_nas"]

    for fname in cfg:
        filltyp, dtyp = cfg[fname]["how"], cfg[fname]["type"]
        s_cnt = df[fname].count()
        fill_cnt = df_cnt - s_cnt
        if filltyp == "zero":
            val = 0
        elif filltyp == "mode":
            val = df[fname].dropna().value_counts().idxmax()
        elif filltyp == "median":
            val = df[fname].dropna().quantile()
        elif filltyp == "mean":
            val = df[fname].dropna().mean()
        elif filltyp == "drop":
            df = df.dropna(subset=[fname])
        else:
            assert 0, "Fill type not found!"
        print("Filling column {} with value {} ({} values)".format(
            fname, val, fill_cnt))
        df[fname] = df[fname].fillna(val).astype(dtyp)
    return df


def to_frame(tbl, join_tbls, cfg, additional_columns=[]):
    """
    Leverage all the built in functionality of the sim framework to join to
    the specified tables, only accessing the columns used in cfg (the model
    yaml configuration file), an any additionally passed columns (the sim
    framework is smart enough to figure out which table to grab the column
    off of)

    Parameters
    ----------
    tbl : DataFrameWrapper
        The table to join other tables to
    join_tbls : list of DataFrameWrappers or strs
        A list of tables to join to "tbl"
    cfg : str
        The filename of a yaml configuration file from which to parse the
        strings which are actually used by the model
    additional_columns : list of strs
        A list of additional columns to include

    Returns
    -------
    A single DataFrame with the index from tbl and the columns used by cfg
    and any additional columns specified
    """
    join_tbls = join_tbls if isinstance(join_tbls, list) else [join_tbls]
    tables = [tbl] + join_tbls
    cfg = yaml_to_class(cfg).from_yaml(str_or_buffer=cfg)
    tables = [t for t in tables if t is not None]
    columns = misc.column_list(tables, cfg.columns_used()) + additional_columns
    if len(tables) > 1:
        df = orca.merge_tables(target=tables[0].name,
                               tables=tables, columns=columns)
    else:
        df = tables[0].to_frame(columns)
    check_nas(df)
    return df


def yaml_to_class(cfg):
    """
    Convert the name of a yaml file and get the Python class of the model
    associated with the configuration

    Parameters
    ----------
    cfg : str
        The name of the yaml configuration file

    Returns
    -------
    Nothing
    """
    import yaml
    model_type = yaml.load(open(cfg))["model_type"]
    return {
        "regression": RegressionModel,
        "segmented_regression": SegmentedRegressionModel,
        "discretechoice": MNLDiscreteChoiceModel,
        "segmented_discretechoice": SegmentedMNLDiscreteChoiceModel
    }[model_type]


def hedonic_estimate(cfg, tbl, join_tbls, out_cfg=None):
    """
    Estimate the hedonic model for the specified table

    Parameters
    ----------
    cfg : string
        The name of the yaml config file from which to read the hedonic model
    tbl : DataFrameWrapper
        A dataframe for which to estimate the hedonic
    join_tbls : list of strings
        A list of land use dataframes to give neighborhood info around the
        buildings - will be joined to the buildings using existing broadcasts
    out_cfg : string, optional
        The name of the yaml config file to which to write the estimation
        results. If not given, the input file cfg is overwritten.
    """
    cfg = misc.config(cfg)
    df = to_frame(tbl, join_tbls, cfg)
    if out_cfg is not None:
        out_cfg = misc.config(out_cfg)
    return yaml_to_class(cfg).fit_from_cfg(df, cfg, outcfgname=out_cfg)


def hedonic_simulate(cfg, tbl, join_tbls, out_fname, cast=False):
    """
    Simulate the hedonic model for the specified table

    Parameters
    ----------
    cfg : string
        The name of the yaml config file from which to read the hedonic model
    tbl : DataFrameWrapper
        A dataframe for which to estimate the hedonic
    join_tbls : list of strings
        A list of land use dataframes to give neighborhood info around the
        buildings - will be joined to the buildings using existing broadcasts
    out_fname : string
        The output field name (should be present in tbl) to which to write
        the resulting column to
    cast : boolean
        Should the output be cast to match the existing column.
    """
    cfg = misc.config(cfg)
    df = to_frame(tbl, join_tbls, cfg)
    price_or_rent, _ = yaml_to_class(cfg).predict_from_cfg(df, cfg)
    tbl.update_col_from_series(out_fname, price_or_rent, cast=cast)


def lcm_estimate(cfg, choosers, chosen_fname, buildings, join_tbls,
                 out_cfg=None):
    """
    Estimate the location choices for the specified choosers

    Parameters
    ----------
    cfg : string
        The name of the yaml config file from which to read the location
        choice model
    choosers : DataFrameWrapper
        A dataframe of agents doing the choosing
    chosen_fname : str
        The name of the column (present in choosers) which contains the ids
        that identify the chosen alternatives
    buildings : DataFrameWrapper
        A dataframe of buildings which the choosers are locating in and which
        have a supply.
    join_tbls : list of strings
        A list of land use dataframes to give neighborhood info around the
        buildings - will be joined to the buildings using existing broadcasts
    out_cfg : string, optional
        The name of the yaml config file to which to write the estimation
        results. If not given, the input file cfg is overwritten.
    """
    cfg = misc.config(cfg)
    choosers = to_frame(choosers, [], cfg, additional_columns=[chosen_fname])
    alternatives = to_frame(buildings, join_tbls, cfg)
    if out_cfg is not None:
        out_cfg = misc.config(out_cfg)
    return yaml_to_class(cfg).fit_from_cfg(choosers,
                                           chosen_fname,
                                           alternatives,
                                           cfg,
                                           outcfgname=out_cfg)


def lcm_simulate(cfg, choosers, buildings, join_tbls, out_fname,
                 supply_fname, vacant_fname,
                 enable_supply_correction=None, cast=True):
    """
    Simulate the location choices for the specified choosers

    Parameters
    ----------
    cfg : string
        The name of the yaml config file from which to read the location
        choice model
    choosers : DataFrameWrapper
        A dataframe of agents doing the choosing
    buildings : DataFrameWrapper
        A dataframe of buildings which the choosers are locating in and which
        have a supply
    join_tbls : list of strings
        A list of land use dataframes to give neighborhood info around the
        buildings - will be joined to the buildings using existing broadcasts.
    out_fname : string
        The column name to write the simulated location to
    supply_fname : string
        The string in the buildings table that indicates the amount of
        available units there are for choosers, vacant or not
    vacant_fname : string
        The string in the buildings table that indicates the amount of vacant
        units there will be for choosers
    enable_supply_correction : Python dict
        Should contain keys "price_col" and "submarket_col" which are set to
        the column names in buildings which contain the column for prices and
        an identifier which segments buildings into submarkets
    cast : boolean
        Should the output be cast to match the existing column.
    """
    cfg = misc.config(cfg)

    choosers_df = to_frame(choosers, [], cfg, additional_columns=[out_fname])

    additional_columns = [supply_fname, vacant_fname]
    if (enable_supply_correction is not None
            and "submarket_col" in enable_supply_correction):
        additional_columns += [enable_supply_correction["submarket_col"]]
    if (enable_supply_correction is not None
            and "price_col" in enable_supply_correction):
        additional_columns += [enable_supply_correction["price_col"]]
    locations_df = to_frame(buildings, join_tbls, cfg,
                            additional_columns=additional_columns)

    available_units = buildings[supply_fname]
    vacant_units = buildings[vacant_fname]

    print("There are {:d} total available units\n".format(
        int(available_units.sum())),
        "    and {:d} total choosers\n".format(
            len(choosers)),
        "    but there are {:d} overfull buildings".format(
            len(vacant_units[vacant_units < 0])))

    vacant_units = vacant_units[vacant_units > 0]

    # sometimes there are vacant units for buildings that are not in the
    # locations_df, which happens for reasons explained in the warning below
    indexes = np.repeat(vacant_units.index.values,
                        vacant_units.values.astype('int'))
    isin = pd.Series(indexes).isin(locations_df.index)
    missing = len(isin[isin == False])  # noqa
    indexes = indexes[isin.values]
    units = locations_df.loc[indexes].reset_index()
    check_nas(units)

    print("    for a total of %d temporarily empty units" % vacant_units.sum())
    print("    in %d buildings total in the region" % len(vacant_units))

    if missing > 0:
        print("WARNING: %d indexes aren't found in the locations df -" %
              missing)
        print("    this is usually because of a few records that don't join ")
        print("    correctly between the locations df",
              "and the aggregations tables")

    movers = choosers_df[choosers_df[out_fname] == -1]
    print("There are %d total movers for this LCM" % len(movers))

    if enable_supply_correction is not None:
        assert isinstance(enable_supply_correction, dict)
        assert "price_col" in enable_supply_correction
        price_col = enable_supply_correction["price_col"]
        assert "submarket_col" in enable_supply_correction
        submarket_col = enable_supply_correction["submarket_col"]

        lcm = yaml_to_class(cfg).from_yaml(str_or_buffer=cfg)

        if enable_supply_correction.get("warm_start", False) is True:
            raise NotImplementedError()

        multiplier_func = enable_supply_correction.get("multiplier_func", None)
        if multiplier_func is not None:
            multiplier_func = orca.get_injectable(multiplier_func)

        kwargs = enable_supply_correction.get('kwargs', {})
        new_prices, submarkets_ratios = supply_and_demand(
            lcm,
            movers,
            units,
            submarket_col,
            price_col,
            base_multiplier=None,
            multiplier_func=multiplier_func,
            **kwargs)

        # we will only get back new prices for those alternatives
        # that pass the filter - might need to specify the table in
        # order to get the complete index of possible submarkets
        submarket_table = enable_supply_correction.get("submarket_table", None)
        if submarket_table is not None:
            submarkets_ratios = submarkets_ratios.reindex(
                orca.get_table(submarket_table).index).fillna(1)
            # write final shifters to the submarket_table for use in debugging
            orca.get_table(submarket_table)[
                "price_shifters"] = submarkets_ratios

        print("Running supply and demand")
        print("Simulated Prices")
        print(buildings[price_col].describe())
        print("Submarket Price Shifters")
        print(submarkets_ratios.describe())
        # we want new prices on the buildings, not on the units, so apply
        # shifters directly to buildings and ignore unit prices
        orca.add_column(buildings.name,
                        price_col + "_hedonic", buildings[price_col])
        new_prices = (buildings[price_col]
                      * submarkets_ratios.loc[buildings[submarket_col]].values)
        buildings.update_col_from_series(price_col, new_prices)
        print("Adjusted Prices")
        print(buildings[price_col].describe())

    if len(movers) > vacant_units.sum():
        print("WARNING: Not enough locations for movers\n",
              "    reducing locations to size of movers for performance gain")
        movers = movers.head(int(vacant_units.sum()))

    new_units, _ = yaml_to_class(cfg).predict_from_cfg(movers, units, cfg)

    # new_units returns nans when there aren't enough units,
    # get rid of them and they'll stay as -1s
    new_units = new_units.dropna()

    # go from units back to buildings
    new_buildings = pd.Series(units.loc[new_units.values][out_fname].values,
                              index=new_units.index)

    choosers.update_col_from_series(out_fname, new_buildings, cast=cast)
    _print_number_unplaced(choosers, out_fname)

    if enable_supply_correction is not None:
        new_prices = buildings[price_col]
        if "clip_final_price_low" in enable_supply_correction:
            new_prices = new_prices.clip(lower=enable_supply_correction[
                "clip_final_price_low"])
        if "clip_final_price_high" in enable_supply_correction:
            new_prices = new_prices.clip(upper=enable_supply_correction[
                "clip_final_price_high"])
        buildings.update_col_from_series(price_col, new_prices)

    vacant_units = buildings[vacant_fname]
    print("    and there are now %d empty units" % vacant_units.sum())
    print("    and %d overfull buildings" % len(
        vacant_units[vacant_units < 0]))


def simple_relocation(choosers, relocation_rate, fieldname, cast=True):
    """
    Run a simple rate based relocation model

    Parameters
    ----------
    choosers : DataFrameWrapper or DataFrame
        Table of agents that might relocate
    relocation_rate : float
        Rate of relocation
    fieldname : str
        The field name in the resulting dataframe to set to -1 (to unplace
        new agents)
    cast : boolean
        Should the output be cast to match the existing column.

    Returns
    -------
    Nothing
    """
    print("Total agents: %d" % len(choosers))
    _print_number_unplaced(choosers, fieldname)

    print("Assigning for relocation...")
    chooser_ids = np.random.choice(choosers.index, size=int(relocation_rate *
                                                            len(choosers)),
                                   replace=False)
    choosers.update_col_from_series(fieldname,
                                    pd.Series(-1, index=chooser_ids),
                                    cast=cast)

    _print_number_unplaced(choosers, fieldname)


def simple_transition(tbl, rate, location_fname):
    """
    Run a simple growth rate transition model on the table passed in

    Parameters
    ----------
    tbl : DataFrameWrapper
        Table to be transitioned
    rate : float
        Growth rate
    location_fname : str
        The field name in the resulting dataframe to set to -1 (to unplace
        new agents)

    Returns
    -------
    Nothing
    """
    transition = GrowthRateTransition(rate)
    df = tbl.to_frame(tbl.local_columns)

    print("%d agents before transition" % len(df.index))
    df, added, copied, removed = transition.transition(df, None)
    print("%d agents after transition" % len(df.index))

    df.loc[added, location_fname] = -1
    orca.add_table(tbl.name, df)
    orca.add_table('new_{}'.format(tbl.name), added)


def full_transition(agents, agent_controls, year, settings, location_fname,
                    linked_tables=None):
    """
    Run a transition model based on control totals specified in the usual
    UrbanSim way

    Parameters
    ----------
    agents : DataFrameWrapper
        Table to be transitioned
    agent_controls : DataFrameWrapper
        Table of control totals
    year : int
        The year, which will index into the controls
    settings : dict
        Contains the configuration for the transition model - is specified
        down to the yaml level with a "total_column" which specifies the
        control total and an "add_columns" param which specified which
        columns to add when calling to_frame (should be a list of the columns
        needed to do the transition
    location_fname : str
        The field name in the resulting dataframe to set to -1 (to unplace
        new agents)
    linked_tables : dict of tuple, optional
        Dictionary of table_name: (table, 'column name') pairs. The column name
        should match the index of `agents`. Indexes in `agents` that
        are copied or removed will also be copied and removed in
        linked tables.

    Returns
    -------
    Nothing
    """
    ct = agent_controls.to_frame()
    hh = agents.to_frame(agents.local_columns +
                         settings.get('add_columns', []))
    print("Total agents before transition: {}".format(len(hh)))
    linked_tables = linked_tables or {}
    for table_name, (table, col) in linked_tables.iteritems():
        print("Total %s before transition: %s" % (table_name, len(table)))
    tran = transition.TabularTotalsTransition(ct, settings['total_column'])
    model = transition.TransitionModel(tran)
    new, added_hh_idx, new_linked = model.transition(
        hh, year, linked_tables=linked_tables)
    new.loc[added_hh_idx, location_fname] = -1
    print("Total agents after transition: {}".format(len(new)))
    orca.add_table(agents.name, new)
    for table_name, table in new_linked.iteritems():
        print("Total %s after transition: %s" % (table_name, len(table)))
        orca.add_table(table_name, table)


def _print_number_unplaced(df, fieldname):
    print("Total currently unplaced: {:d}".format(
        df[fieldname].value_counts().get(-1, 0)))


# INFORMATIONAL OCCUPANCY TABLE
def average_occupancy(agents, buildings, residential):
    """
    Parameters
    ----------
    agents
    buildings
    residential

    Returns
    -------

    """

    agents_per_building = agents.building_id.value_counts()

    if residential:
        occupancy = (agents_per_building
                     / buildings.residential_units)
    else:
        job_sqft_per_building = (agents_per_building
                                 * buildings.sqft_per_job)
        occupancy = (job_sqft_per_building
                     / buildings.non_residential_sqft)

    occupancy = occupancy.clip(upper=1.0)

    return occupancy.mean()


def run_occupancy(year, occupancy, buildings,
                  households, new_households,
                  jobs, new_jobs,
                  sqft_per_job, years_previous):
    """
    Register a DataFrame indexed by year, with uses as columns. Values are
    number of years to absorb existing inventory given yearly demand.

    Parameters
    ----------
    year
    occupancy
    buildings
    households
    new_households
    jobs
    new_jobs
    sqft_per_job : numeric or Series
    years_previous : int

    Returns
    -------

    """

    occupancy = occupancy.to_frame()
    building_columns = ['residential_units',
                        'non_residential_sqft',
                        'year_built']

    buildings = buildings.to_frame(building_columns)
    buildings['sqft_per_job'] = sqft_per_job

    starting_year = year - years_previous
    buildings = buildings.loc[buildings.year_built >= starting_year]

    for use in ['residential', 'non_residential']:

        residential = True if use == 'residential' else False
        agents = households if use == 'residential' else jobs
        # new_agents = new_households if use == 'residential' else new_jobs

        if use == 'residential':
            submarket = buildings.loc[buildings.residential_units > 0]
        else:
            submarket = buildings.loc[buildings.non_residential_sqft > 0]

        occ = average_occupancy(agents, submarket, residential=residential)
        occupancy.loc[year, use] = occ

    orca.add_table('occupancy', occupancy)
    return occupancy


def prepare_parcels_for_feasibility(parcels, parcel_price_callback,
                                    pf, parcel_occupancy_callback=None,
                                    start_year=None, years_back=20):
    """
    Prepare parcel DataFrame for feasibility analysis

    Parameters
    ----------
    parcels : DataFrame Wrapper
        The data frame wrapper for the parcel data
    parcel_price_callback : function
        A callback which takes each use of the pro forma and returns a series
        with index as parcel_id and value as yearly_rent
    pf: SqFtProForma object
        Pro forma object with relevant configurations
    parcel_occupancy_callback : function
        A callback which takes each use of the pro forma, along with a start
        year, and returns series with index as parcel_id and value as
        expected occupancy
    start_year : int
        Year to begin tracking occupancy
    years_back : int
        Occupancy will be calculated for buildings built starting in this year

    Returns
    -------
    DataFrame of parcels
    """

    df = parcels.to_frame()

    if pf.parcel_filter:
        df = df.query(pf.parcel_filter)

    current_year = orca.get_injectable('year')
    oldest_year = current_year - years_back

    for use in pf.uses:
        # Add prices
        df[use] = parcel_price_callback(use)

        # Add occupancies
        if start_year and current_year >= start_year:
            occ_col = 'occ_{}'.format(use)
            df[occ_col] = parcel_occupancy_callback(use, oldest_year)

    # convert from cost to yearly rent
    if pf.residential_to_yearly and 'residential' in df.columns:
        df["residential"] *= pf.cap_rate

    # df = occupancy_regional(df, pf, start_year)

    return df


def lookup_by_form(df, parcel_use_allowed_callback, pf, **kwargs):
    """
    Execute development feasibility on all parcels

    Parameters
    ----------
    df : DataFrame Wrapper
        The data frame wrapper for the parcel data
    parcel_use_allowed_callback : function
        A callback which takes each use of the pro forma and returns a series
        with index as parcel_id and value as yearly_rent
    pf: SqFtProForma object
        Pro forma object with relevant configurations

    Returns
    -------
    DataFrame of parcels
    """

    lookup_results = {}

    forms = pf.forms_to_test or pf.forms
    for form in forms:
        print("Computing feasibility for form %s" % form)
        allowed = parcel_use_allowed_callback(form).loc[df.index]

        newdf = df[allowed]

        lookup_results[form] = pf.lookup(form, newdf, **kwargs)

    feasibility = pd.concat(lookup_results.values(),
                            keys=lookup_results.keys(),
                            axis=1)

    return feasibility


def run_feasibility(parcels, parcel_price_callback,
                    parcel_use_allowed_callback,
                    parcel_occupancy_callback=None, start_year=None,
                    years_back=20, cfg=None, **kwargs):
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
    years_back : int
        Number of years back to track occupancy for
    cfg : str
        The name of the yaml file to read pro forma configurations from

    Returns
    -------
    Adds a table called feasibility to the sim object (returns nothing)
    """

    cfg = misc.config(cfg)
    pf = (sqftproforma.SqFtProForma.from_yaml(str_or_buffer=cfg) if cfg
          else sqftproforma.SqFtProForma.from_defaults())
    df = prepare_parcels_for_feasibility(parcels, parcel_price_callback,
                                         pf, parcel_occupancy_callback,
                                         start_year, years_back)
    feasibility = lookup_by_form(df, parcel_use_allowed_callback, pf, **kwargs)
    orca.add_table('feasibility', feasibility)


def _remove_developed_buildings(old_buildings, new_buildings, unplace_agents):
    redev_buildings = old_buildings.parcel_id.isin(new_buildings.parcel_id)
    l1 = len(old_buildings)
    drop_buildings = old_buildings[redev_buildings]

    if "dropped_buildings" in orca.orca._TABLES:
        prev_drops = orca.get_table("dropped_buildings").to_frame()
        orca.add_table("dropped_buildings",
                       pd.concat([drop_buildings, prev_drops]))
    else:
        orca.add_table("dropped_buildings", drop_buildings)

    old_buildings = old_buildings[np.logical_not(redev_buildings)]
    l2 = len(old_buildings)
    if l2 - l1 > 0:
        print("Dropped {} buildings because they were redeveloped".format(
            l2 - l1))

    for tbl in unplace_agents:
        agents = orca.get_table(tbl).local
        displaced_agents = agents.building_id.isin(drop_buildings.index)
        print("Unplaced {} before: {}".format(tbl,
                                              len(agents.query(
                                                  "building_id == -1"))))
        agents.building_id[displaced_agents] = -1
        print("Unplaced {} after: {}".format(tbl,
                                             len(agents.query(
                                                 "building_id == -1"))))

    return old_buildings


def process_new_buildings(feasibility, buildings, new_buildings,
                          form_to_btype_callback,
                          add_more_columns_callback,
                          supply_fname, remove_developed_buildings,
                          unplace_agents):
    if form_to_btype_callback is not None:
        new_buildings["building_type_id"] = new_buildings.apply(
            form_to_btype_callback, axis=1)

    ret_buildings = new_buildings
    if add_more_columns_callback is not None:
        new_buildings = add_more_columns_callback(new_buildings)

    print("Adding {:,} buildings with {:,} {}".format(
        len(new_buildings),
        int(new_buildings[supply_fname].sum()),
        supply_fname))

    print("{:,} feasible buildings after running developer".format(
        len(feasibility)))

    old_buildings = buildings.to_frame(buildings.local_columns)
    new_buildings = new_buildings[buildings.local_columns]

    if remove_developed_buildings:
        old_buildings = _remove_developed_buildings(
            old_buildings, new_buildings, unplace_agents)

    all_buildings, new_index = merge_buildings(old_buildings, new_buildings,
                                               return_index=True)
    ret_buildings.index = new_index

    return all_buildings, ret_buildings


def add_new_units(dev, new_buildings):
    config = dev.to_dict

    if "residential_units" in orca.list_tables() and config['residential']:
        # need to add units to the units table as well
        old_units = orca.get_table("residential_units")
        old_units = old_units.to_frame(old_units.local_columns)

        unit_num = np.concatenate(
            [np.arange(i) for i in new_buildings.residential_units.values])

        building_id = np.repeat(
            new_buildings.index.values,
            new_buildings.residential_units.astype('int32').values)

        new_units = pd.DataFrame({
            "unit_residential_price": 0,
            "num_units": 1,
            "deed_restricted": 0,
            "unit_num": unit_num,
            "building_id": building_id
        })

        new_units.sort(columns=["building_id", "unit_num"], inplace=True)
        new_units.reset_index(drop=True, inplace=True)

        print("Adding {:,} units to the residential_units table".format(
            len(new_units)))
        all_units = merge_buildings(old_units, new_units)
        all_units.index.name = "unit_id"

        orca.add_table("residential_units", all_units)


def compute_units_to_build(num_agents, num_units, target_vacancy):
    """
    Compute number of units to build to match target vacancy.

    Parameters
    ----------
    num_agents : int
        number of agents that need units in the region
    num_units : int
        number of units in buildings
    target_vacancy : float (0-1.0)
        target vacancy rate

    Returns
    -------
    number_of_units : int
        the number of units that need to be built
    """
    print("Number of agents: {:,}".format(num_agents))
    print("Number of agent spaces: {:,}".format(int(num_units)))
    assert target_vacancy < 1.0
    target_units = int(max(
        (num_agents / (1 - target_vacancy) - num_units), 0))
    print("Current vacancy = {:.2f}".format(1 - num_agents /
                                            float(num_units)))
    print("Target vacancy = {:.2f}, target of new units = {:,}".format(
        target_vacancy,
        target_units))
    return target_units


def merge_buildings(old_df, new_df, return_index=False):
    """
    Merge two dataframes of buildings.  The old dataframe is
    usually the buildings dataset and the new dataframe is a modified
    (by the user) version of what is returned by the pick method.

    Parameters
    ----------
    old_df : dataframe
        Current set of buildings
    new_df : dataframe
        New buildings to add, usually comes from this module
    return_index : bool
        If return_index is true, this method will return the new
        index of new_df (which changes in order to create a unique
        index after the merge)

    Returns
    -------
    df : dataframe
        Combined DataFrame of buildings, makes sure indexes don't overlap
    index : pd.Index
        If and only if return_index is True, return the new index for the
        new_df dataframe (which changes in order to create a unique index
        after the merge)
    """
    maxind = np.max(old_df.index.values)
    new_df = new_df.reset_index(drop=True)
    new_df.index = new_df.index + maxind + 1
    concat_df = pd.concat([old_df, new_df], verify_integrity=True)
    concat_df.index.name = 'building_id'

    if return_index:
        return concat_df, new_df.index

    return concat_df


def run_developer(forms, agents, buildings, supply_fname, feasibility,
                  parcel_size, ave_unit_size, current_units, cfg, year=None,
                  target_vacancy=0.1, form_to_btype_callback=None,
                  add_more_columns_callback=None,
                  remove_developed_buildings=True,
                  unplace_agents=['households', 'jobs'],
                  num_units_to_build=None, profit_to_prob_func=None,
                  min_profit_per_sqft=None):
    """
    Run the developer model to pick and build buildings

    Parameters
    ----------
    forms : string or list of strings
        Passed directly dev.pick
    agents : DataFrame Wrapper
        Used to compute the current demand for units/floorspace in the area
    buildings : DataFrame Wrapper
        Used to compute the current supply of units/floorspace in the area
    supply_fname : string
        Identifies the column in buildings which indicates the supply of
        units/floorspace
    feasibility : DataFrame Wrapper
        The output from feasibility above (the table called 'feasibility')
    parcel_size : series
        The size of the parcels.  This was passed to feasibility as well,
        but should be passed here as well.  Index should be parcel_ids.
    ave_unit_size : series
        The average residential unit size around each parcel - this is
        indexed by parcel, but is usually a disaggregated version of a
        zonal or accessibility aggregation.
    current_units : series
        The current number of units on the parcel.  Is used to compute the
        net number of units produced by the developer model.  Many times
        the developer model is redeveloping units (demolishing them) and
        is trying to meet a total number of net units produced.
    cfg : str
        The name of the yaml file to read pro forma configurations from
    year : int
        The year of the simulation - will be assigned to 'year_built' on the
        new buildings
    target_vacancy : float
        The target vacancy rate - used to determine how much to build
    form_to_btype_callback : function
        Will be used to convert the 'forms' in the pro forma to
        'building_type_id' in the larger model
    add_more_columns_callback : function
        Takes a dataframe and returns a dataframe - is used to make custom
        modifications to the new buildings that get added
    remove_developed_buildings : optional, boolean (default True)
        Remove all buildings on the parcels which are being developed on
    unplace_agents : optional, list of strings (default ['households', 'jobs'])
        For all tables in the list, will look for field building_id and set
        it to -1 for buildings which are removed - only executed if
        remove_developed_buildings is true
    num_units_to_build: optional, int
        If num_units_to_build is passed, build this many units rather than
        computing it internally by using the length of agents adn the sum of
        the relevant supply columin - this trusts the caller to know how to
        compute this.
    profit_to_prob_func: func
        Passed directly to dev.pick
    min_profit_per_sqft: optional, numeric
        If passed, developer model will build all buildings that have a profit
        per square foot larger than this value (subject to other constraints)

    Returns
    -------
    Writes the result back to the buildings table and returns the new
    buildings with available debugging information on each new building
    """
    cfg = misc.config(cfg)

    if min_profit_per_sqft:
        target_units = None
    else:
        target_units = (
            num_units_to_build or
            compute_units_to_build(len(agents),
                                   buildings[supply_fname].sum(),
                                   target_vacancy))

    dev = develop.Developer.from_yaml(feasibility.to_frame(), forms,
                                      target_units, parcel_size,
                                      ave_unit_size, current_units,
                                      year, str_or_buffer=cfg)

    print("{:,} feasible buildings before running developer".format(
        len(dev.feasibility)))

    new_buildings = dev.pick(profit_to_prob_func, min_profit_per_sqft)
    orca.add_table("feasibility", dev.feasibility)

    if new_buildings is None:
        return

    if len(new_buildings) == 0:
        return new_buildings

    all_buildings, ret_buildings = (
        process_new_buildings(dev.feasibility, buildings, new_buildings,
                              form_to_btype_callback,
                              add_more_columns_callback,
                              supply_fname, remove_developed_buildings,
                              unplace_agents))

    add_new_units(dev, ret_buildings)

    orca.add_table("buildings", all_buildings)

    return ret_buildings


def scheduled_development_events(buildings, new_buildings,
                                 remove_developed_buildings=True,
                                 unplace_agents=['households', 'jobs']):
    """
    This acts somewhat like developer, but is not dependent on real estate
    feasibility in order to build - these are buildings that we force to be
    built, usually because we know they are scheduled to be built at some
    point in the future because of our knowledge of existing permits
    (or maybe we just read the newspaper).

    Parameters
    ----------
    buildings : DataFrame wrapper
        Just pass in the building dataframe wrapper
    new_buildings : DataFrame
        The new buildings to add to out buildings table.  They should have the
        same columns as the local columns in the buildings table.
    """

    print("Adding {:,} buildings as scheduled development events".format(
        len(new_buildings)))

    old_buildings = buildings.to_frame(buildings.local_columns)
    new_buildings = new_buildings[buildings.local_columns]

    print("Res units before: {:,}".format(
        old_buildings.residential_units.sum()))
    print("Non-res sqft before: {:,}".format(
        old_buildings.non_residential_sqft.sum()))

    if remove_developed_buildings:
        old_buildings = \
            _remove_developed_buildings(old_buildings, new_buildings,
                                        unplace_agents)

    all_buildings = develop.Developer.merge(old_buildings, new_buildings)

    print("Res units after: {:,}".format(
        all_buildings.residential_units.sum()))
    print("Non-res sqft after: {:,}".format(
        all_buildings.non_residential_sqft.sum()))

    orca.add_table("buildings", all_buildings)
    return new_buildings


def check_store_for_bytes(store):
    """
    Checks HDF5 stores for column names or index names that are bytes types
    rather than strings, which can break references in a model

    Parameters
    ----------
    store : str
        Path to HDF5 store to check for bytes types

    Returns
    -------
    found_byte : bool
        True if a bytes types is found in a column name or index name
    """
    found_byte = False
    byte_types = (np.bytes_, bytes)
    with pd.HDFStore(store) as store:
        for name in store.keys():
            table = store[name]
            if table.index.name and type(table.index.name) in byte_types:
                found_byte = True
            for col in table.columns:
                if type(col) in byte_types:
                    found_byte = True
    return found_byte


def decode_byte_df(df):
    """
    Converts bytes types to strings in index names and column names
    of a DataFrame

    Parameters
    ----------
    df : DataFrame
        Input DataFrame to convert

    Returns
    -------
    df : DataFrame
        Output DataFrame, with converted index names and column names

    """
    byte_types = (np.bytes_, bytes)

    if isinstance(df.index, pd.core.index.MultiIndex):
        new_names = [name.decode() if type(name) in byte_types else name
                     for name in df.index.names]
        df.index.set_names(new_names, inplace=True)

    elif df.index.name and type(df.index.name) in byte_types:
        df.index.name = df.index.name.decode()

    df.columns = [col.decode() if type(col) in byte_types else col
                  for col in df.columns]

    return df
