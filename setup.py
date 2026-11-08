from setuptools import setup, find_packages

requirements = [
    'numpy',
    'scipy',
    'matplotlib',
    'pyyaml',
    'appdirs',
    # Not on PyPI; installed from GitHub. An existing install (e.g. an
    # editable checkout of vedb-gaze) satisfies these.
    'file_io @ git+https://github.com/piecesofmindlab/file_io.git',
    'vedb_gaze @ git+https://github.com/vedb/vedb-gaze.git',
    ]

setup(
    name='vedb_store',
    version='0.0.1',
    packages=find_packages(),
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    install_requires=requirements,
    include_package_data=True,
    package_data={
        'vedb_store':[
          'defaults.cfg',
            ],
        },    
)
