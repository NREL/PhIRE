from setuptools import setup, find_packages

console_scripts = [
    'rplearn-data=phire.rplearn.data_tool:main',
    'rplearn-train=phire.rplearn.train:main',
    'phire-train=phire.main:main',
    'phire-eval=phire.evaluation.cli:main',
    'phire-data=phire.era5_to_tfrecords:main'
]

packages = find_packages('python')

setup(
    name='PhIRE',
    version='2.0',
    packages=packages,
    package_dir = {'': 'python'},
    entry_points = {
        'console_scripts': console_scripts
    }
)
