# UrbanSim Parcel-level Starter Model

[![Build Status](https://travis-ci.com/urbansim/urbansim_parcels.svg?token=GSDNqBio5uUExRqdD5zJ&branch=master)](https://travis-ci.com/urbansim/urbansim_parcels)

This repository provides a starting place for parcel level UrbanSim models, and
provides two working example models, with public data and configurations
included in the repo itself. This version is a replacement for both
`urbansim_defaults` and `sanfran_urbansim`, and works with the new UDST
developer model. Works with Python 2.7 and 3.5.

The examples provided here are discussed in detail in the documentation for the new [developer model](https://urbansim.github.io/developer/).

### Requirements
* urbansim
* orca
* pandana
* osmnet
* developer

### Optional
A good and strongly recommended practice is the use of an isolated Python environment where to
run every project, install development and pre-release versions of packages and avoid
conflicts with other Python projects running on the same machine.

* Python virtual environment (virtualenv)
    To install: `pip install virtualenv`

### Installation steps
We recommend running UrbanSim models in a virtual environment through conda or virtualenv.
Conda virtual environment setup can follow along with steps in the Travis build file (.travis.yml).

```
git clone https://github.com/urbansim/urbansim_parcels
cd urbansim_parcels
python setup.py install
```

The San Francisco model is a simple example:
```
cd urbansim_parcels/sf_example
python simulate.py
```

The San Diego example has a more fully-featured set of models. We've included a
small subset of the San Diego data and network in this repo. The full data can
be downloaded at [this link](https://dl.dropboxusercontent.com/u/69619688/sandiego_data_12042015.zip).

Run the subset model:
```
cd urbansim_parcels/sd_example
python Simulation.py
```

Run the full model:
```
cd urbansim_parcels/sd_example/data
curl -O https://dl.dropboxusercontent.com/u/69619688/sandiego_data_12042015.zip
unzip sandiego_data_12042015.zip
# replace the "store" and "net_store" variables with the correct names in settings.yaml
cd ..
python Simulation.py
```

Running the simulation scripts off the bat relies on pre-estimated
configurations; you can also run the estimation workflows using
`estimate.py` or `Estimation.py` in each example. Note: Estimation on San Diego
region should be done using the full dataset, rather than the subset.