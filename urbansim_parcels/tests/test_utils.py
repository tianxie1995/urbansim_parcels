import pandas as pd
import numpy as np
from urbansim_parcels import utils
import pytest


def test_developer_compute_units_to_build():
    to_build = utils.compute_units_to_build(30, 30, .1)
    assert int(to_build) == 3


def test_developer_merge():
    df1 = pd.DataFrame({'test': [1]}, index=[1])
    df2 = pd.DataFrame({'test': [1]}, index=[1])
    merge = utils.merge_buildings(df1, df2)
    # make sure index is unique
    assert merge.index.values[1] == 2


@pytest.fixture
def df_byte_col():
    data = {
        b'a': [1, 2, 3],
        'b': [1, 2, 3],
        b'c': [1, 2, 3]
    }
    index = [0, 1, 2]
    return pd.DataFrame(data, index)


@pytest.fixture
def df_byte_col():
    data = {
        b'a': [1, 2, 3],
        'b': [1, 2, 3],
        b'c': [1, 2, 3]
    }
    index = [0, 1, 2]
    return pd.DataFrame(data, index)


@pytest.fixture
def df_byte_index_name():
    data = {
        'a': [1, 2, 3],
        'b': [1, 2, 3],
        'c': [1, 2, 3]
    }
    index = [0, 1, 2]
    df = pd.DataFrame(data, index)
    df.index.name = b'index'
    return df


@pytest.fixture
def byte_store(df_byte_col, df_byte_index_name, tmpdir):
    p = tmpdir.mkdir('data').join('byte_store.h5')
    store = pd.HDFStore(str(p), 'w')
    store.put('df_byte_col', df_byte_col)
    store.put('df_byte_index_name', df_byte_index_name)
    store.close()
    return str(p)


def test_check_for_bytes(byte_store):

    assert utils.check_store_for_bytes(byte_store) is True


def test_decode_bytes(df_byte_col, df_byte_index_name, tmpdir):

    decoded_df_byte_col = utils.decode_byte_df(df_byte_col)
    decoded_df_byte_index_name = utils.decode_byte_df(df_byte_index_name)

    p = tmpdir.mkdir('data').join('byte_store.h5')
    store = pd.HDFStore(str(p), 'w')
    store.put('decoded_df_byte_col', decoded_df_byte_col)
    store.put('decoded_df_byte_index_name', decoded_df_byte_index_name)
    store.close()

    assert utils.check_store_for_bytes(str(p)) is False
