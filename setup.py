from setuptools import setup, find_packages

setup(
    name='urbansim_parcels',
    version='0.1dev',
    description='Urbansim parcel model',
    author='UrbanSim Inc.',
    author_email='info@urbansim.com',
    url='https://github.com/urbansim/urbansim_parcels',
    classifiers=[
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.5'
        'Programming Language :: Python :: 3.6'
        'Programming Language :: Python :: 3.7'
    ],
    packages=find_packages(exclude=['*.tests']),
    install_requires=[
        'developer',  # install manually: https://github.com/urbansim/developer
        'numpy',
        'orca',
        'pandana',
        'pandas',
        'urbansim',
    ]
)