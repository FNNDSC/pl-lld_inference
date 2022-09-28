from os import path
from setuptools import setup

with open(path.join(path.dirname(path.abspath(__file__)), 'README.rst')) as f:
    readme = f.read()

setup(
    name             = 'lld_inference',
    version          = '1.0.4',
    description      = 'An app to run LLD inference',
    long_description = readme,
    author           = 'FNNDSC',
    author_email     = 'dev@babyMRI.org',
    url              = 'http://wiki',
    packages         = ['lld_inference',
                        'LLDcode',
                        'LLDcode.datasets',
                        'LLDcode.graph',
                        'LLDcode.datasources',
                        'LLDcode.generators',
                        'LLDcode.iterators',
                        'LLDcode.tensorflow_train',
                        'LLDcode.tensorflow_train.utils',
                        'LLDcode.tensorflow_train.layers',
                        'LLDcode.tensorflow_train.losses',
                        'LLDcode.tensorflow_train.networks',
                        'LLDcode.utils',
                        'LLDcode.utils.io',
                        'LLDcode.utils.landmark',
                        'LLDcode.transformations',
                        'LLDcode.transformations.spatial',
                        'LLDcode.transformations.intensity.np',
                        'LLDcode.transformations.intensity.sitk'],
    install_requires = ['chrisapp'],
    test_suite       = 'nose.collector',
    tests_require    = ['nose'],
    license          = 'MIT',
    zip_safe         = False,
    python_requires  = '>=3.6',
    entry_points     = {
        'console_scripts': [
            'lld_inference = lld_inference.__main__:main'
            ]
        }
)
