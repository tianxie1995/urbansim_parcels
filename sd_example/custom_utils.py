from __future__ import print_function, division, absolute_import

import json

import orca
import pandas as pd
from urbansim.utils import misc


class SimulationSummaryData(object):
    """
    Keep track of zone-level and parcel-level output for use in the
    simulation explorer.  Writes the correct format and filenames that the
    simulation explorer expects.

    Parameters
    ----------
    run_number : int
        The run number for this run
    zone_indicator_file : optional, str
        A template for the zone_indicator_filename - use {} notation and the
        run_number will be substituted.  Should probably not be modified if
        using the simulation explorer.
    parcel_indicator_file : optional, str
        A template for the parcel_indicator_filename - use {} notation and the
        run_number will be substituted.  Should probably not be modified if
        using the simulation explorer.
    """

    def __init__(self,
                 run_number,
                 zone_indicator_file="runs/run{}_simulation_output.json",
                 parcel_indicator_file="runs/run{}_parcel_output.csv"):
        self.run_num = run_number
        self.zone_indicator_file = zone_indicator_file.format(run_number)
        self.parcel_indicator_file = \
            parcel_indicator_file.format(run_number)
        self.parcel_output = None
        self.zone_output = None

    def add_zone_output(self, zones_df, name, year, round=2):
        """
        Pass in a dataframe and this function will store the results in the
        simulation state to write out at the end (to describe how the
        simulation changes over time)

        Parameters
        ----------
        zones_df : DataFrame
            dataframe of indicators whose index is the zone_id and columns are
            indicators describing the simulation
        name : string
            The name of the dataframe to use to differentiate all the sources
            of the indicators
        year : int
            The year to associate with these indicators
        round : int
            The number of decimal places to round to in the output json

        Returns
        -------
        Nothing
        """
        # this creates a hierarchical json data structure to encapsulate
        # zone-level indicators over the simulation years.  "index" is the ids
        # of the shapes that this will be joined to and "year" is the list of
        # years. Each indicator then get put under a two-level dictionary of
        # column name and then year.  this is not the most efficient data
        # structure but since the number of zones is pretty small, it is a
        # simple and convenient data structure
        if self.zone_output is None:
            d = {
                "index": list(zones_df.index),
                "years": []
            }
        else:
            d = self.zone_output

        assert d["index"] == (list(zones_df.index),
                              "Passing in zones dataframe that is not aligned"
                              "on the same index as a previous dataframe")

        if year not in d["years"]:
            d["years"].append(year)

        for col in zones_df.columns:
            d.setdefault(col, {})
            d[col]["original_df"] = name
            s = zones_df[col]
            dtype = s.dtype
            if dtype == "float64" or dtype == "float32":
                s = s.fillna(0)
                d[col][year] = [float(x) for x in list(s.round(round))]
            elif dtype == "int64" or dtype == "int32":
                s = s.fillna(0)
                d[col][year] = [int(x) for x in list(s)]
            else:
                d[col][year] = list(s)

        self.zone_output = d

    def add_parcel_output(self, new_parcel_output):
        """
        Add new parcel-level indicators to the parcel output.

        Parameters
        ----------
        new_parcel_output : DataFrame
            Adds a new set of parcel data for output exploration - this data
            is merged with previous data that has been added.  This data is
            generally used to capture new developments that UrbanSim has
            predicted, thus it doesn't override previous years' indicators

        Returns
        -------
        Nothing
        """
        if new_parcel_output is None:
            return

        if self.parcel_output is not None:
            # merge with old  parcel output
            self.parcel_output = (
                pd.concat([self.parcel_output, new_parcel_output])
                .reset_index(drop=True))
        else:
            self.parcel_output = new_parcel_output

    def write_parcel_output(self,
                            add_xy=None):
        """
        Write the parcel-level output to a csv file

        Parameters
        ----------
        add_xy : dictionary (optional)
            Used to add x, y values to the output - an example dictionary is
            pasted below - the parameters should be fairly self explanatory.
            Note that from_epsg and to_epsg can be omitted in which case the
            coordinate system is not changed.  NOTE: pyproj is required
            if changing coordinate systems::

                {
                    "xy_table": "parcels",
                    "foreign_key": "parcel_id",
                    "x_col": "x",
                    "y_col": "y",
                    "from_epsg": 3740,
                    "to_epsg": 4326
                }


        Returns
        -------
        Nothing
        """
        if self.parcel_output is None:
            return

        po = self.parcel_output
        if add_xy is not None:
            x_name, y_name = add_xy["x_col"], add_xy["y_col"]
            xy_joinname = add_xy["foreign_key"]
            xy_df = orca.get_table(add_xy["xy_table"])
            po[x_name] = misc.reindex(xy_df[x_name], po[xy_joinname])
            po[y_name] = misc.reindex(xy_df[y_name], po[xy_joinname])

            if "from_epsg" in add_xy and "to_epsg" in add_xy:
                import pyproj
                p1 = pyproj.Proj('+init=epsg:%d' % add_xy["from_epsg"])
                p2 = pyproj.Proj('+init=epsg:%d' % add_xy["to_epsg"])
                x2, y2 = pyproj.transform(p1, p2,
                                          po[x_name].values,
                                          po[y_name].values)
                po[x_name], po[y_name] = x2, y2

        po.to_csv(self.parcel_indicator_file, index_label="development_id")

    def write_zone_output(self):
        """
        Write the zone-level output to a file.
        """
        if self.zone_output is None:
            return
        outf = open(self.zone_indicator_file, "w")
        json.dump(self.zone_output, outf)
        outf.close()


def building_occupancy(oldest_year=None):
    """
    Add "occupancy" column to buildings table using units for residential and
    square footage for nonresidential uses. Used in occupancy_vars custom
    model for San Diego.

    Parameters
    ----------
    oldest_year : int, optional
        If passed, buildings built before oldest_year will be filtered out

    Returns
    -------
    buildings : DataFrame
    """
    households, jobs, buildings = ([orca.get_table(table) for table in
                                    ['households', 'jobs', 'buildings']])

    buildings = (buildings
                 .to_frame(['parcel_id', 'residential_units',
                            'non_residential_sqft', 'job_spaces',
                            'zone_id', 'year_built', 'general_type',
                            'node_id']))

    buildings = (buildings[buildings.year_built >= oldest_year]
                 if oldest_year is not None else buildings)

    buildings['sqft_per_job'] = (buildings.non_residential_sqft
                                 / buildings.job_spaces)

    # Residential
    households = households.to_frame(columns=['building_id'])
    agents_per_building = households.building_id.value_counts()
    buildings['occupancy_res'] = ((agents_per_building
                                   / buildings.residential_units)
                                  .clip(upper=1.0))

    # Non-residential
    jobs = jobs.to_frame(columns=['building_id'])
    agents_per_building = jobs.building_id.value_counts()
    job_sqft_per_building = (agents_per_building
                             * buildings.sqft_per_job)
    buildings['occupancy_nonres'] = ((job_sqft_per_building
                                      / buildings.non_residential_sqft)
                                     .clip(upper=1.0))

    return buildings
