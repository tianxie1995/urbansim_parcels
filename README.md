# UrbanSim Parcels Model
Aimed to easily setup an experimentation environment on simulations at parcel level
using the ProForma development module.

Replaces `urbansim_defaults` for compatibility with the new UrbanSim Developer Model.

This kit is preconfigured to use with San Francisco City.

### Requirements (Linux OS)
* git
* wget
* Python 2.7+
* pip (Python package manager)
* Python virtual environments
    To install: `pip install virtualenv`

### Installation steps

* Clone this repo
    ```
    https://github.com/urbansim/urbansim_parcels
    cd urbansim_parcels
    ```
* Create Python virtual environment
    ```
    virtualenv venv
    ```
* Activate virtual environment
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
* Exit Python virtual environment
    ```
    deactivate
    ```

## Usage
* Activate existing Python environment
    ```
    source venv/bin/activate
    ```
* Run San Francisco example simulation
    ```
    cd urbansim_parcels/sf_example
    python simulate.py
    ```
