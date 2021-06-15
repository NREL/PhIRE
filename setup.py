from setuptools import setup, find_packages

console_scripts = [
    'rplearn-data=phire.rplearn.data_tool:main',
    'rplearn-train=phire.rplearn.train:main'
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
