from setuptools import setup, find_packages

requirements = [
    'numpy', 'scipy', 'couchdb', ] # file_io, docdb_lite

setup(
    name='vedb_store',
    version='0.0.1',
    packages=find_packages(),
    long_description=open('README.md').read(),
    install_requires=requirements,
    include_package_data=True,
    package_data={
        'vedb_store':[
          'defaults.cfg',
            ],
        },    
)
