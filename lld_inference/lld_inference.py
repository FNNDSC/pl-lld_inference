#
# lld_inference ds ChRIS plugin app
#
# (c) 2022 Fetal-Neonatal Neuroimaging & Developmental Science Center
#                   Boston Children's Hospital
#
#              http://childrenshospital.org/FNNDSC/
#                        dev@babyMRI.org
#

import  logging
import  warnings
warnings.filterwarnings(action='ignore',message='Python 3.6 is no longer supported')

import  os
from    chrisapp.base import ChrisApp
from    LLDcode.main import MainLoop
import  glob
import  pudb

Gstr_title = r"""
 _ _     _   _        __
| | |   | | (_)      / _|
| | | __| |  _ _ __ | |_ ___ _ __ ___ _ __   ___ ___
| | |/ _` | | | '_ \|  _/ _ \ '__/ _ \ '_ \ / __/ _ \
| | | (_| | | | | | | ||  __/ | |  __/ | | | (_|  __/
|_|_|\__,_| |_|_| |_|_| \___|_|  \___|_| |_|\___\___|
        ______
       |______|
"""

Gstr_synopsis = """


    NAME

       lld_inference

    SYNOPSIS

        docker run --rm fnndsc/pl-lld_inference lld_inference           \\
            [-f|--inputFileFilter <inputFileFilter>]                    \\
            [--heatmapThreshold <f_fraction>]                           \\
            [-h] [--help]                                               \\
            [--json]                                                    \\
            [--man]                                                     \\
            [--meta]                                                    \\
            [--savejson <DIR>]                                          \\
            [-v |--verbosity <level>]                                   \\
            [--version]                                                 \\
            <inputDir>                                                  \\
            <outputDir>

    BRIEF EXAMPLE

        * Bare bones execution

            docker run --rm -u $(id -u)                                 \\
                -v $(pwd)/in:/incoming -v $(pwd)/out:/outgoing          \\
                fnndsc/pl-lld_inference lld_inference                   \\
                /incoming /outgoing

    DESCRIPTION

        `lld_inference` is a *ChRIS ds-type* application that works off
        inputs of leg images (X-Ray images converted to `*mha` format).
        For each image, the plugin attempts to determine where six landmarks
        are located -- three in each leg (in layman's terms):

            * top (or superior) of femur bone (close to the hip around the
              intertrochanteric line)
            * middle of the knee, slightly inferior to the intercondylar notch
            * bottom (or inferior) of the tibial bone (just inferior to the
              medial malleolus)

        For each inferred location of these three landmark in each leg, the
        plugin generates a heatmap image for that landmark. Thus, six separate
        images are created -- blank everywhere except for the heatmap. Taken
        all together, these six images are then superimposed onto the original
        image for comparison. In addition to output images, the plugin also
        generates a text file in `csv` format that describes the heatmap
        centroids in `(x, y)` coordinate pairs.

    ARGS
        [-f|--inputFileFilter <inputFileFilter>]
        A glob pattern string, default is "**/*.mha", representing the input
        file pattern to analyze.

        [--heatmapThreshold <f_fraction>]
        A fractional value between 0.0 and 1.0 that defines the threshold for
        heatmap cutoff. Values in calculated inference heatmaps greater than
        <f_fraction> are conserved, while all others are set to zero. This
        reduces image noise considerably (compare heatmaps as generated in
        the original 'inferenceSpace' vs those filtered and transformed to
        the 'referenceSpace').

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
"""

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)

class Lld_inference(ChrisApp):
    """
    An app to run LLD inference.
    """
    PACKAGE                 = __package__
    TITLE                   = 'A ChRIS plugin that runs an inference model to predict landmark points on leg images '
    CATEGORY                = ''
    TYPE                    = 'ds'
    ICON                    = ''   # url of an icon image
    MIN_NUMBER_OF_WORKERS   = 1    # Override with the minimum number of workers as int
    MAX_NUMBER_OF_WORKERS   = 1    # Override with the maximum number of workers as int
    MIN_CPU_LIMIT           = 3000 # Override with millicore value as int (1000 millicores == 1 CPU core)
    MIN_MEMORY_LIMIT        = 32000  # Override with memory MegaByte (MB) limit as int
    MIN_GPU_LIMIT           = 1    # Override with the minimum number of GPUs as int
    MAX_GPU_LIMIT           = 1    # Override with the maximum number of GPUs as int

    OUTPUT_META_DICT = {}

    def define_parameters(self):
        """
        Define the CLI arguments accepted by this plugin app.
        Use self.add_argument to specify a new app argument.
        """
        self.add_argument(  '--inputFileFilter','-f',
                            dest         = 'inputFileFilter',
                            type         = str,
                            optional     = True,
                            help         = 'Input file filter',
                            default      = '**/*.mha')

        self.add_argument(  '--heatmapThreshold',
                            dest         = 'heatmapThreshold',
                            type         = str,
                            optional     = True,
                            help         = 'fractional heatmap threshold',
                            default      = '0.5')

        self.add_argument(  '--compositeWeight',
                            dest         = 'compositeWeight',
                            type         = str,
                            optional     = True,
                            help         = 'heatmap-over-reference weighting',
                            default      = '0.3,0.7')

        self.add_argument(  '--heatmapKernel',
                            dest         = 'heatmapKernel',
                            type         = str,
                            optional     = True,
                            help         = 'size of heatmap kernel in reference space',
                            default      = '7')

        self.add_argument(  '--imageType',
                            dest         = 'imageType',
                            type         = str,
                            optional     = True,
                            help         = 'output image type',
                            default      = 'jpg')

    def NVIDIA_scan(self):
        """
        Simply show the VGA devices reported by `lspci` as a very basic check
        on GPU resources
        """
        print('------------------------------------------------------')
        print('Scanning devices using lspci...')
        print('------------------------------------------------------')
        os.system('lspci | grep -i VGA')
        print('------------------------------------------------------')
        print('Checking for nvidia drivers...')
        print('------------------------------------------------------')
        os.system('lsmod | grep nvidia')
        print('------------------------------------------------------')
        print('Starting inference...')
        print('------------------------------------------------------')

    def run(self, options):
        """
        Define the code to be run by this plugin app.
        """
        print(Gstr_title)
        print('Version: %s' % self.get_version())

        # Output the space of CLI
        d_options = vars(options)
        for k,v in d_options.items():
            print("%20s: %-40s" % (k, v))
        print("")

        self.NVIDIA_scan()

        dataset_path    = options.inputdir
        str_glob        = '%s/%s' % (options.inputdir, options.inputFileFilter)
        l_datapath      = glob.glob(str_glob, recursive=True)
        dataset_path    = os.path.dirname(l_datapath[0])
        # pudb.set_trace()
        MainLoop.run(
            dataset_path,
            options.outputdir,
            float(options.heatmapThreshold),
            options.heatmapKernel,
            options.compositeWeight,
            options.imageType
        )

    def show_man_page(self):
        """
        Print the app's man page.
        """
        print(Gstr_synopsis)
