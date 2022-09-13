#
# lld_inference ds ChRIS plugin app
#
# (c) 2022 Fetal-Neonatal Neuroimaging & Developmental Science Center
#                   Boston Children's Hospital
#
#              http://childrenshospital.org/FNNDSC/
#                        dev@babyMRI.org
#

from chrisapp.base import ChrisApp
from LLDcode.main import MainLoop


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

        docker run --rm fnndsc/pl-lld_inference lld_inference                     \\
            [-h] [--help]                                               \\
            [--json]                                                    \\
            [--man]                                                     \\
            [--meta]                                                    \\
            [--savejson <DIR>]                                          \\
            [-v <level>] [--verbosity <level>]                          \\
            [--version]                                                 \\
            <inputDir>                                                  \\
            <outputDir> 

    BRIEF EXAMPLE

        * Bare bones execution

            docker run --rm -u $(id -u)                             \
                -v $(pwd)/in:/incoming -v $(pwd)/out:/outgoing      \
                fnndsc/pl-lld_inference lld_inference                        \
                /incoming /outgoing

    DESCRIPTION

        `lld_inference` ...

    ARGS

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


class Lld_inference(ChrisApp):
    """
    An app to run LLD inference .
    """
    PACKAGE                 = __package__
    TITLE                   = 'A ChRIS plugin that runs an inference model to predict landmark points on leg images '
    CATEGORY                = ''
    TYPE                    = 'ds'
    ICON                    = ''   # url of an icon image
    MIN_NUMBER_OF_WORKERS   = 1    # Override with the minimum number of workers as int
    MAX_NUMBER_OF_WORKERS   = 1    # Override with the maximum number of workers as int
    MIN_CPU_LIMIT           = 2000 # Override with millicore value as int (1000 millicores == 1 CPU core)
    MIN_MEMORY_LIMIT        = 8000  # Override with memory MegaByte (MB) limit as int
    MIN_GPU_LIMIT           = 0    # Override with the minimum number of GPUs as int
    MAX_GPU_LIMIT           = 0    # Override with the maximum number of GPUs as int

    # Use this dictionary structure to provide key-value output descriptive information
    # that may be useful for the next downstream plugin. For example:
    #
    # {
    #   "finalOutputFile":  "final/file.out",
    #   "viewer":           "genericTextViewer",
    # }
    #
    # The above dictionary is saved when plugin is called with a ``--saveoutputmeta``
    # flag. Note also that all file paths are relative to the system specified
    # output directory.
    OUTPUT_META_DICT = {}

    def define_parameters(self):
        """
        Define the CLI arguments accepted by this plugin app.
        Use self.add_argument to specify a new app argument.
        """
        self.add_argument(  '--inputDirFilter','-d',
                            dest         = 'inputDirFilter',
                            type         = str,
                            optional     = True,
                            help         = 'Input directory filter',
                            default      = 'EOS_dataset')

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
        
        
        st_glob = ""
        if len(options.inputDirFilter):
            str_glob = '%s/%s' % (options.inputdir, options.inputDirFilter)
        
        if len(str_glob):
            dir_hits = glob.glob(str_glob, recursive = True)
            
            
        if len(dir_hits): dataset_path = dir_hits[0]

        MainLoop.run(dataset_path,options.outputdir)

    def show_man_page(self):
        """
        Print the app's man page.
        """
        print(Gstr_synopsis)
