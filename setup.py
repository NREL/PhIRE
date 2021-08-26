from setuptools import setup, find_packages

console_scripts = [
    'rplearn-data=phire.data_tool:rplearn_main',
    'rplearn-train=phire.rplearn.train:main',
    'phire-train=phire.main:main',
    'phire-eval=phire.evaluation.cli:main',
    'phire-data=phire.data_tool:phire_main'
]


package_data = {
    'phire.jetstream': ['data/*.csv']
}

packages = find_packages('python')


setup(
    name='PhIRE',
    version='2.0',
    packages=packages,
    package_dir = {'': 'python'},
    package_data=package_data,
    entry_points = {
        'console_scripts': console_scripts
    }
)
