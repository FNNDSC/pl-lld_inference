from setuptools import setup


setup(
    name             = 'lld_inference',
    version          = '2.2.12',
    description      = 'An app to run LLD inference',
    author           = 'FNNDSC',
    author_email     = 'dev@babyMRI.org',
    url              = 'https://github.com/FNNDSC/pl-lld_inference#readme',
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
