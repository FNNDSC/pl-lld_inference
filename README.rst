pl-lld_inference
================================

.. image:: https://img.shields.io/docker/v/fnndsc/pl-lld_inference?sort=semver
    :target: https://hub.docker.com/r/fnndsc/pl-lld_inference

.. image:: https://img.shields.io/github/license/fnndsc/pl-lld_inference
    :target: https://github.com/FNNDSC/pl-lld_inference/blob/master/LICENSE

.. image:: https://github.com/FNNDSC/pl-lld_inference/workflows/ci/badge.svg
    :target: https://github.com/FNNDSC/pl-lld_inference/actions


.. contents:: Table of Contents


Abstract
--------

This plugin application is part of the Leg-Length-Discrepancy project that measures leg lengths based off X-Ray images. This particular plugin operates on input `mha` format images, and infers the locations of several anatomical landmarks on the leg images. These locations are shown as heat maps (i.e. small fuzzy regions of probalities) that are superimposed on the original images. In addition, a prediction text file in `csv` format with the locations of the heatmaps is also generated.


Description
-----------


``lld_inference`` is a *ChRIS ds-type* application that works off inputs of leg images (X-Ray images converted to `*mha` format). For each image, the plugin attempts to determine where six landmarks are located -- three in each leg (in layman's terms):

* top (or superior) of femur bone (close to the hip around the intertrochanteric line)
* middle of the knee, slightly inferior to the intercondylar notch
* bottom (or inferior) of the tibial bone (just inferior to the medial malleolus)

For each inferred location of these three landmark in each leg, the plugin generates a heatmap image for that landmark. Thus, six separate images are created -- blank everywhere except for the heatmap. Taken all together, these six images are then superimposed onto the original image for comparison. In addition to output images, the plugin also generates a text file in `csv` format that describes the heatmap centroids in `(x, y)` coordinate pairs.

Usage
-----

.. code::

    docker run --rm fnndsc/pl-lld_inference lld_inference
        [-f|--inputFileFilter <inputFileFilter>]
        [-h|--help]
        [--json] [--man] [--meta]
        [--savejson <DIR>]
        [-v|--verbosity <level>]
        [--version]
        <inputDir> <outputDir>


Arguments
~~~~~~~~~

.. code::

    [-f|--inputFileFilter <inputFileFilter>]
    A glob pattern string, default is "**/*.mha", representing the input
    file pattern to analyze.

    [-h] [--help]
    If specified, show help message and exit.

    [--json]
    If specified, show json representation of app and exit.

    [--man]
    If specified, print (this) man page and exit.

    [--meta]
    If specified, print plugin meta data and exit.

    [--savejson <DIR>]
    If specified, save json representation file to DIR and exit.

    [-v <level>] [--verbosity <level>]
    Verbosity level for app. Not used currently.

    [--version]
    If specified, print version number and exit.


Getting inline help is:

.. code:: bash

    docker run --rm fnndsc/pl-lld_inference lld_inference --man

Run
~~~

You need to specify input and output directories using the `-v` flag to
`docker run`. Also set the output directory to be world writable


.. code:: bash

    mkdir in out && chmod 777 out
    docker run --rm -u $(id -u)                             \
        -v $PWD/in:/incoming -v $PWD/out:/outgoing          \
        fnndsc/pl-lld_inference lld_inference               \
        /incoming /outgoing


Development
-----------

Build the Docker container:

.. code:: bash

    docker build -t local/pl-lld_inference .

Run unit tests:

.. code:: bash

    docker run --rm local/pl-lld_inference nosetests

For in-container debugging, mount the source directories appropriately:


.. code:: bash
    
    cd pl-lld_inference
    docker run -it --rm                                                             \
        -v $PWD/LLDcode:/opt/conda/lib/python3.6/site-packages/LLDcode              \
        -v $PWD/lld_inference:/opt/conda/lib/python3.6/site-packages/lld_inference  \
        -v $PWD/in:/incoming -v $PWD/out:/outgoing                                  \
        local/pl-lld_inference lld_inference                                        \
        /incoming /outgoing

Examples
--------

.. code:: bash

    # Assume you have some *mha leg images... copy them to the input directory.
    # Obviously adjust below as you see fit!
    cd ~/some/dir
    mkdir in out && chmod 777 out
    cp *mha in
    docker run --rm -u $(id -u)                             \
        -v $PWD/in:/incoming -v $PWD/out:/outgoing          \
        fnndsc/pl-lld_inference lld_inference               \
        /incoming /outgoing

_-30-_

.. image:: https://raw.githubusercontent.com/FNNDSC/cookiecutter-chrisapp/master/doc/assets/badge/light.png
    :target: https://chrisstore.co
