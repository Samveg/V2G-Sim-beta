try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

config = {
    'description': 'Vehicle to grid simulator',
    'author': 'Samveg Saxena, Jonathan Coignard',
    'author_email': 'jcoignard@lbl.gov',
    'version': '0.1',
    'install_requires': ['nose'],
    'packages': ['v2gsim', 'v2gsim.driving', 'v2gsim.charging',
                 'v2gsim.post_simulation', 'v2gsim.driving.drivecycle',
                 'v2gsim.driving.detailed', 'v2gsim.battery_degradation'],
    'name': 'v2gsim',
    'package_data': {'': ['*.mat']},  # Add the drivecycle matlab data
    'include_package_data': True,
}

setup(**config)
