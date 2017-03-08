from urbansim_parcels import utils


def test_developer_compute_units_to_build():
    to_build = utils._compute_units_to_build(30, 30, .1)
    assert int(to_build) == 3
