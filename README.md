# UrbanSim Parcel-level Starter Model
This repository provides a starting place for parcel level UrbanSim models, and provides two working example models, with public data and configurations included in the repo itself. This version is a replacement for both `urbansim_defaults` and `sanfran_urbansim`, and works with the new UDST developer model. Works with Python 2.7 and 3.5.

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
We recommend running UrbanSim models in a virtual environment through conda or virtualenv. Conda virtual environment setup can follow along with steps in the Travis build file (.travis.yml).

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

The San Diego example has a more fully-featured set of models. We've included a small subset of the San Diego data and network in this repo. The full data can be downloaded at [this link](https://dl.dropboxusercontent.com/u/69619688/sandiego_data_12042015.zip).

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
