# UrbanSim Parcels Model
Aimed to easily setup an experimentation environment on simulations at parcel level
using the ProForma development module.

Replaces `urbansim_defaults` for compatibility with the new UrbanSim Developer Model.

This kit is preconfigured to use with San Francisco City.

### Requirements (Linux OS)
* git
* Python 2.7
* pip (Python package manager)

### Optional
A good and strongly recommended practice is the use of an isolated Python environment where to
run every project, install development and pre-release versions of packages and avoid
conflicts with other Python projects running on the same machine.

* Python virtual environment (virtualenv)
    To install: `pip install virtualenv`

### Installation steps

* Clone this repo
    ```
    https://github.com/urbansim/urbansim_parcels
    cd urbansim_parcels
    ```
* (only when using `virtualenv`) Create Python virtual environment
    ```
    virtualenv venv
    ```
* (`venv` only) Activate virtual environment
    ```
    source venv/bin/activate
    ```
* Install dependencies
    ```
    pip install numpy pandas
    ```
* Install UrbanSim Stack
    ```
    # Pandana
    pip install pandana
    # UrbanSim
    pip install https://github.com/UDST/urbansim/archive/master.zip
    # Orca
    pip install https://github.com/UDST/orca/archive/master.zip
    # ProForma Developer module
    git clone git@github.com:urbansim/developer.git
    pip install ./developer
    # Parcels models
    pip install .
    ```
* (`venv` only) Exit Python virtual environment
    ```
    deactivate
    ```

## Usage
* (`venv` only) Activate existing Python environment
    ```
    source venv/bin/activate
    ```
* Run San Francisco example simulation
    ```
    cd urbansim_parcels/sf_example
    python simulate.py
    ```
