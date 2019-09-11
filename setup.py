try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

__authors__ = ['Anselm Hahn']
__license__ = 'MIT'
__version__ = '0.5'
__date__ = '11/09/2019'

with open('requirements.txt') as f:
    required = f.read().splitlines()


setup(
	name='pyinsights',
	python_requires='>3.6.0',
	version=__version__,
	packages=['test', 'examples', 'pyinsights','docs'],
	package_data={'examples': ['*csv'],'docs': ['*csv']},
	include_package_data=True,
	install_requires=required,
	url='https://github.com/Anselmoo/csv_first_insight',
	download_url='https://github.com/Anselmoo/TanabeSugano/releases',
	license=__license__,
	author=__authors__,
	author_email='Anselm.Hahn@gmail.com',
	description='A sklearn-based correlation- and prediction-maker for small csv-data with below 10,000 entries.',
	platforms=['MacOS :: MacOS X', 'Microsoft :: Windows','POSIX :: Linux'],
)
