import pandas as pd
from urbansim_parcels import utils


def test_developer_compute_units_to_build():
    to_build = utils.compute_units_to_build(30, 30, .1)
    assert int(to_build) == 3


def test_developer_merge():
    df1 = pd.DataFrame({'test': [1]}, index=[1])
    df2 = pd.DataFrame({'test': [1]}, index=[1])
    merge = utils.merge_buildings(df1, df2)
    # make sure index is unique
    assert merge.index.values[1] == 2
