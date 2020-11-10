# flexnoise setup script

from setuptools import setup

setup(
    name='flexnoise',
    description='Flexible noise processes for time series',
    version='0.1a',
    install_requires=[
        'matplotlib>=3.2',
        'numpy>=1.17',
        'pints',
        'pytest',
        'scipy>=1.3'
    ]
)
