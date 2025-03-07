# Install setuptools if not installed.
try:
    import setuptools
except ImportError:
    from ez_setup import use_setuptools
    use_setuptools()

from setuptools import setup, find_packages

setup(
    name='urbansim_parcels',
    version='0.1.dev1',
    description='Urbansim parcel model',
    author='UrbanSim Inc.',
    author_email='info@urbansim.com',
    url='https://github.com/urbansim/urbansim_parcels',
    classifiers=[
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.5'
    ],
    packages=find_packages(exclude=['*.tests']),
    install_requires=[
        'numpy >= 1.1.0',
        'pandas >= 0.16.0',
        'orca >= 1.3.0',
        'urbansim >= 0.1.1',
        'developer'
    ],
    extras_require={
        'pandana': ['pandana>=0.1']
    }
)
