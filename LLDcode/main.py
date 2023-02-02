
import os, sys
import numpy as np
import tensorflow as tf
from collections import OrderedDict

import  LLDcode.tensorflow_train
import  LLDcode.tensorflow_train.utils.tensorflow_util
from    LLDcode.tensorflow_train.utils.data_format import get_batch_channel_image_size
import  LLDcode.utils
import  LLDcode.utils.io.image
import  LLDcode.utils.io.landmark
import  LLDcode.utils.io.text
from    LLDcode.tensorflow_train.data_generator     import  DataGenerator
from    LLDcode.tensorflow_train.train_loop         import  MainLoopBase
from    LLDcode.utils.landmark.heatmap_test         import  HeatmapTest

from datetime import datetime
import time

from .dataset import Dataset
from .network import network_scn, network_unet, network_downsampling, network_conv, network_scn_mmwhs

import  itertools
from    sklearn.preprocessing                       import  normalize
from    PIL                                         import  Image as pImage
import  cv2
import  imageio
import  SimpleITK                                   as      sitk
from    SimpleITK                                   import  Image
from    pathlib                                     import  Path
from    functools                                   import  reduce
from    operator                                    import  mul
from    scipy                                       import  interpolate
import  skimage.filters

import  pudb
import  tempfile

from    loguru                  import logger
LOG             = logger.debug

logger_format = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> │ "
    "<level>{level: <5}</level> │ "
    "<yellow>{name: >28}</yellow>::"
    "<cyan>{function: <30}</cyan> @"
    "<cyan>{line: <4}</cyan> ║ "
    "<level>{message}</level>"
)
logger.remove()
logger.opt(colors = True)
logger.add(sys.stderr, format=logger_format)



class MainLoop(MainLoopBase):
    def __init__(
                    self,
                    cv, network_id, inputdir, outputdir,
                    heatmapThreshold, heatmapKernel, compositeWeight,
                    imageType
        ):
        super().__init__()
        self.cv                     = cv
        self.network_id             = network_id
        self.output_folder          = outputdir
        self.batch_size             = 8
        self.learning_rate          = 0.0000001 # TODO adapt learning rates for different networks for faster training
        self.max_iter               = 20000
        self.test_iter              = 5000
        self.disp_iter              = 10
        self.snapshot_iter          = self.test_iter
        self.test_initialization    = True
        self.current_iter           = 0
        self.reg_constant           = 0.00005
        self.invert_transformation  = False
        self.heatmapThreshold       = heatmapThreshold
        self.heatmapKernel          = heatmapKernel
        self.compositeWeight        = compositeWeight
        self.imageType              = imageType

        image_sizes = {'scn': [256, 128],
                       'unet': [256, 256],
                       'downsampling': [256, 256],
                       'conv': [256, 128],
                       'scn_mmwhs': [256, 256]}

        heatmap_sizes = {'scn': [256, 128],
                         'unet': [256, 256],
                         'downsampling': [64, 64],
                         'conv': [256, 128],
                         'scn_mmwhs': [256, 256]}

        sigmas = {'scn': 3.0,
                  'unet': 3.0,
                  'downsampling': 1.5,
                  'conv': 2,
                  'scn_mmwhs': 3.0}

        self.image_size = image_sizes[self.network_id]
        self.heatmap_size = heatmap_sizes[self.network_id]
        self.sigma = sigmas[self.network_id]
        self.image_channels = 1
        self.num_landmarks = 6
        self.data_format = 'channels_first'
        self.save_debug_images = False
        self.base_folder = inputdir
        dataset = Dataset(self.image_size,
                          self.heatmap_size,
                          self.num_landmarks,
                          self.sigma,
                          self.base_folder,
                          self.cv,
                          self.data_format,
                          self.save_debug_images)

        self.dataset_val = dataset.dataset_val()
        self.dataset_train =  self.dataset_val
        networks = {'scn': network_scn,
                    'unet': network_unet,
                    'downsampling': network_downsampling,
                    'conv': network_conv,
                    'scn_mmwhs': network_scn_mmwhs}
        self.network = networks[self.network_id]
        self.loss_function = lambda x, y: tf.nn.l2_loss(x - y) / get_batch_channel_image_size(x, self.data_format)[0]
        self.files_to_copy = []

    def initNetworks(self):
        net = tf.make_template('net', self.network)
        if self.data_format == 'channels_first':
            data_generator_entries = OrderedDict([('image', [self.image_channels] + list(reversed(self.image_size))),
                                                  ('landmarks', [self.num_landmarks] + list(reversed(self.heatmap_size)))])
            data_generator_entries_val = OrderedDict([('image', [self.image_channels] + list(reversed(self.image_size))),
                                                      ('landmarks', [self.num_landmarks] + list(reversed(self.heatmap_size)))])
        else:
            raise NotImplementedError
        self.train_queue = DataGenerator(self.dataset_train, self.coord, data_generator_entries, batch_size=self.batch_size, n_threads=8)
        placeholders = self.train_queue.dequeue()
        image = placeholders[0]
        landmarks = placeholders[1]
        prediction = net(image, num_landmarks=self.num_landmarks, is_training=True, data_format=self.data_format)
        self.loss_net = self.loss_function(landmarks, prediction)

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            if self.reg_constant > 0:
                reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
                self.loss_reg = self.reg_constant * tf.add_n(reg_losses)
                self.loss = self.loss_net + self.loss_reg
            else:
                self.loss_reg = 0
                self.loss = self.loss_net

        self.train_losses = OrderedDict([('loss', self.loss_net), ('loss_reg', self.loss_reg)])

        # build val graph
        self.val_placeholders = LLDcode.tensorflow_train.utils.tensorflow_util.create_placeholders(data_generator_entries_val, shape_prefix=[1])
        self.image_val = self.val_placeholders['image']
        self.landmarks_val = self.val_placeholders['landmarks']
        self.prediction_val = net(self.image_val, num_landmarks=self.num_landmarks, is_training=False, data_format=self.data_format)

        # losses
        self.loss_val = self.loss_function(self.landmarks_val, self.prediction_val)
        self.val_losses = OrderedDict([('loss', self.loss_val), ('loss_reg', self.loss_reg)])

    def test_full_image(self, dataset_entry):
        generators = dataset_entry['generators']
        transformations = dataset_entry['transformations']
        feed_dict = {self.val_placeholders['image']: np.expand_dims(generators['image'], axis=0),
                     self.val_placeholders['landmarks']: np.expand_dims(generators['landmarks'], axis=0)}
        # run loss and update loss accumulators
        run_tuple = self.sess.run((self.prediction_val, self.loss_val) + self.val_loss_aggregator.get_update_ops(), feed_dict)
        prediction = np.squeeze(run_tuple[0], axis=0)
        image = generators['image']
        transformation = transformations['image']
        return image, prediction, transformation

    def inference_do(self):
        heatmap_test    = HeatmapTest( channel_axis=0,
                                    invert_transformation=self.invert_transformation)
        landmarks       = {}
        tic             = time.perf_counter()
        # pudb.set_trace()
        for i in range(self.dataset_val.num_entries()):
            dataset_entry                   = self.dataset_val.get_next()
            current_id                      = dataset_entry['id']
            LOG(f"\nProcessing {current_id}")
            datasources                     = dataset_entry['datasources']
            reference_image                 = datasources['image_datasource']
            image, prediction, transform    = self.test_full_image(dataset_entry)
            LLDcode.utils.io.image.write_np((prediction * 128).astype(np.int8), self.output_file_for_current_iteration(current_id + '_heatmap.mha'))
            predicted_landmarks             = heatmap_test.get_landmarks(prediction, reference_image, transformation=transform)
            # pudb.set_trace()
            p2r_transform(  reference       = reference_image,
                            landmarks       = predicted_landmarks,
                            predictions     = prediction,
                            transform       = transform,
                            heatmapobj      = heatmap_test,
                            heatmapfilter   = self.heatmapThreshold,
                            heatmapkernel   = self.heatmapKernel,
                            composite       = self.compositeWeight,
                            outputdir       = self.output_file_for_current_iteration(current_id),
                            imagetype       = self.imageType).run()
            LLDcode.tensorflow_train.utils.tensorflow_util.print_progress_bar(i, self.dataset_val.num_entries())
            landmarks[current_id]           = predicted_landmarks

        toc=time.perf_counter()
        LLDcode.tensorflow_train.utils.tensorflow_util.print_progress_bar(self.dataset_val.num_entries(), self.dataset_val.num_entries())
        LLDcode.utils.io.landmark.save_points_csv(landmarks, self.output_file_for_current_iteration('prediction.csv'))
        LOG(f"total execute time duration = {toc-tic:4.4f} seconds")

    def run(inputdir, outputdir,
            heatmapThreshold, heatmapKernel, compositeWeight,
            imageType):
        loop        = MainLoop(0, 'conv', inputdir, outputdir,
                                heatmapThreshold, heatmapKernel, compositeWeight,
                                imageType)
        loop.process()

class p2r_transform:
    """
    A class to encapsulate operations relating to transforming a
    "p"redicted_image to a "r"eference image.

    This class implements a "reactive" solution to an observed
    mismatch between heatmaps in the learning space and the transformed
    center coordinate in the reference space.

    Heatmaps (i.e. landmark probabilities) are generated in a 256x128
    image space. While the calculated transformation of the "brightest"
    heatmap point back into the reference space is correct, the heatmap
    itself, if naively mapped back into the reference space, does not
    always match the landmark location.

    This class provides methods to intensity-filter an image and apply
    the same landmark transform to each filtered non-zero pixel.

    """

    def __init__(self, *args, **kwargs):
        self.itk_imageReference     : Image         = None
        self.nd_image               : np.ndarray    = None
        self.lnd_heatmap            : list          = []
        self.lnd_heatmapHPF         : list          = []
        self.nd_heatmapsAll         : np.ndarray    = None
        self.l_landmarks            : list          = []
        self.f_heatmapHighPassFilter: float         = 0.4
        self.itk_transform                          = None
        self.lnd_heatmapRefSpace    : list          = []
        self.path_outputDir         : Path          = Path(tempfile.mkdtemp(prefix='lld-'))
        self.compositeWeight        : tuple         = None
        self.heatmapKernel          : int           = 0
        self.imageType              : str           = 'jpg'

        self.o_heatmap                              = None

        self.mmFilter               = sitk.MinimumMaximumImageFilter()

        for k,v in kwargs.items():
            if k == 'reference'     : self.itk_imageReference       = v
            if k == 'landmarks'     : self.l_landmarks              = v
            if k == 'predictions'   : self.lnd_heatmap              = v
            if k == 'heatmapfilter' : self.f_heatmapHighPassFilter  = v
            if k == 'transform'     : self.itk_transform            = v
            if k == 'heatmapobj'    : self.o_heatmap                = v
            if k == 'outputdir'     : self.path_outputDir           = Path(v)
            if k == 'heatmapkernel' : self.heatmapKernel            = int(v)
            if k == 'composite'     : self.compositeWeight          = tuple(float(s) for s in v.split(','))
            if k == 'imagetype'     : self.imageType                = v

        self.nd_image               = sitk.GetArrayFromImage(self.itk_imageReference)
        self.nd_imageInt            = (self.nd_image*128).astype(np.uint8)

        nd_heatmapRefSpace          = np.zeros(self.nd_image.shape)
        for newheatmap in self.lnd_heatmap:
            self.lnd_heatmapRefSpace.append(nd_heatmapRefSpace.copy())

    @property
    def heatmapFilter(self):
        return self.f_heatmapHighPassFilter

    @heatmapFilter.setter
    def heatmapFilter(self, f_v):
        self.f_heatmapHighPassFilter = f_v

    def imagelist_intensityFilter(self, lnd_image : list, **kwargs) -> dict:
        """
        Simple filter on intensity values in the list of supplied image
        matrices. By default, a high pass filter is performed (i.e. only
        intensity values higher than the thresholdPercentage are allowed
        through). This can be set to a low pass filter with the

                        kwarg: filter = 'lowpass'

        The filter threshold is a percentage of the maximum intensity value
        in the image. By default this will be the heatmapFilter() value, but
        can be overriden with the

                        kwarg: thresholdPerc = <v>

        A new filtered heatmap list is returned in a dictionary, as well as
        some counts on pre/post filter pixel counts.
        """
        l_image                 : list  = []
        l_filtered              : list  = []
        lnd_imageFiltered       : list  = []
        nd_imageFiltered                = None
        l_maxIntensity_coordVal : list  = []
        l_prefilterNonZero      : list  = []
        l_postFilterNonZero     : list  = []
        l_thresholdAbsolute     : list  = []
        index                   : int   = 0

        # pudb.set_trace()

        f_filterThreshold       : float = self.heatmapFilter
        str_filterType          : str   = "highpass"

        for k, v in kwargs.items():
            if k == 'filter'        : str_filterType        = v
            if k == 'thresholdPerc' : f_filterThreshold     = v
            if k == 'thresholdList' : l_thresholdAbsolute   = v

        for nd_image in lnd_image:
            LOG("FILTERING: %s intensity filter on image index %d, shape %s, threshold %f..." %
                    (str_filterType.lower(), index, nd_image.shape, self.heatmapFilter))
            value, coord        = LLDcode.utils.np_image.\
                                    find_quadratic_subpixel_maximum_in_image(nd_image)
            l_prefilterNonZero.append(np.count_nonzero(nd_image))
            f_filterAbsolute    = f_filterThreshold * value
            if len(l_thresholdAbsolute):
                f_filterAbsolute    = l_thresholdAbsolute[index]
            l_image             = nd_image.tolist()
            if 'high' in str_filterType.lower():
                l_filtered          = [[x if x >= f_filterAbsolute else 0.0
                                            for x in y] for y in l_image]
            else:
                l_filtered          = [[x if x <= f_filterAbsolute else 0.0
                                            for x in y] for y in l_image]
            nd_imageFiltered  = np.array(l_filtered)
            l_postFilterNonZero.append(np.count_nonzero(nd_imageFiltered))
            lnd_imageFiltered.append(nd_imageFiltered)
            index            += 1
        return {
            'prefilterNonZero'  :   l_prefilterNonZero,
            'postfilterNonZero' :   l_postFilterNonZero,
            'images'            :   lnd_imageFiltered
        }

    def pointInGrid(self, A_point, a_gridSize, *args):
        """
        SYNOPSIS

            [A_point] = pointInGrid(A_point, a_gridSize [, ab_wrapGridEdges])

        ARGS

            INPUT
            A_point        array of N-D points     points in grid space
            a_gridSize     array                   the size (rows, cols) of
                                                + the grid space

            OPTIONAL
            ab_wrapGridEdges        bool          if True, wrap "external"
                                                points back into grid

            OUTPUT
            A_point        array of N-D points     points that are within the
                                                grid.

        DESC
            Determines if set of N-dimensionals <A_point> is within a grid
            <a_gridSize>.

        PRECONDITIONS
            o Assumes strictly positive domains, i.e. points with negative
            locations are by definition out-of-range. If negative domains
            are valid in a particular problem space, the caller will need
            to offset <a_point> to be strictly positive first.

        POSTCONDITIONS
            o if <ab_wrapGridEdges> is False, returns only the subset of points
            in A_point that are within the <a_gridSize>.
            o if <ab_wrapGridEdges> is True, "wraps" any points in A_point
            back into a_gridSize first, and then checks for those that
            are still within <a_gridSize>.

        """
        b_wrapGridEdges = False # If True, wrap around edges of grid

        if len(args): b_wrapGridEdges = args[0]

        # Check for points "less than" grid space
        if b_wrapGridEdges:
            W = np.where(A_point < 0)
            A_point[W] += a_gridSize[W[1]]

        Wbool = A_point >= 0
        W = Wbool.prod(axis=1)
        A_point = A_point[np.where(W > 0)]

        # Check for points "more than" grid space
        A_inGrid = a_gridSize - A_point
        if b_wrapGridEdges:
            W = np.where(A_inGrid <= 0)
            A_point[W] = -A_inGrid[W]
            A_inGrid = a_gridSize - A_point

        Wbool = A_inGrid > 0
        W = Wbool.prod(axis=1)
        A_point = A_point[np.where(W > 0)]
        return A_point

    def neighbours_findFast(self, a_dimension, a_depth, *args, **kwargs):
        """

            SYNOPSIS

                [A] = neighbours_findFast(a_dimension, a_depth, *args, **kwargs)

            ARGS

            INPUT
            a_dimension         int32           number of dimensions
            a_depth             int32           depth of neighbours to find

            OPTIONAL
            av_origin           array          row vector that defines the
                                                + origin in the <a_dimension>
                                                + space. Neighbours' locations
                                                + are returned relative to this
                                                + origin. Default is the zero
                                                + origin.

                                                If specified, this *must* be
                                                a 1xa_dimension nparray, i.e.
                                                np.array( [(x,y)] ) in the
                                                2D case.

            OUTPUT
            A                   array           a single array containing
                                                    all the neighbours. This
                                                    does not include the origin.

            Named keyword args
            "includeOrigin"     If True, return the origin in the set.
            "wrapGridEdges"     If True, wrap around the edges of grid.
                                If False, do not wrap around grid edges.
            "gridSize"          Specifies the gridSize for 'wrapEdges'.

            If "gridSize" is set, and "wrapEdges" is not, then the neighbors will
            not include coordinates "outside" of the grid.

            DESC
                This method determines the neighbours of a point in an
                n - dimensional discrete space. The "depth" (or ply) to
                calculate is `a_depth'.

            NOTE
                o Uses the 'itertools' module product() method for
                    MUCH faster results than the explicit neighbours_find()
                    method.

            PRECONDITIONS
                o The underlying problem is discrete.

            POSTCONDITIONS
                o Returns all neighbours

            HISTORY
            23 December 2011
            o Rewrote earlier method using python idioms and resulting
            in multiple order speed improvement.
        """

        # Process *kwargs and behavioural arguments
        b_includeOrigin = False # If True, include the "origin" in A
        b_wrapGridEdges = False # If True, wrap around edges of grid
        b_gridSize      = False # Helper flag for tracking wrap
        b_flipAxes      = False # If True, 'flip' axes

        for key, value in kwargs.items():
            if key == 'includeOrigin':      b_includeOrigin = value
            if key == 'wrapGridEdges':      b_wrapGridEdges = value
            if key == 'flipAxes':           b_flipAxes      = value
            if key == 'gridSize':
                a_gridSize = value
                b_gridSize = True

        # Check on *args, i.e. an optional 'origin' point in N-space
        v_origin = np.zeros((a_dimension))
        b_origin = 0
        if len(args):
            av_origin = args[0].astype(np.uint16)
            if b_flipAxes:
                v_origin = np.flip(av_origin)
            else:
                v_origin = av_origin
            b_origin = 1

        A = np.array(list(itertools.product(np.arange(-a_depth, a_depth + 1),
                                            repeat=a_dimension)))
        if not b_includeOrigin:
            Wbool = A == 0
            W = Wbool.prod(axis=1)
            A = A[np.where(W == 0)]
        if b_origin:
            try:
                A += v_origin
            except:
                return np.array(list(v_origin))
        if b_gridSize: A  = self.pointInGrid(A, a_gridSize, b_wrapGridEdges)
        return A

    def heatmaplist_transformToReferenceImageSpace(self, d_filteredHeatMap) -> dict:
        """
        Transform the heatmaps into a new matrix/array in the coordinate system
        (size) of the reference image.

        This re-uses existing LLD code, lightly copy/pasted and factored into
        this class so to have minimal impact on existing code.

        Note that this simply "transforms/projects" the pixels in the low
        resolution heatmap into an ndarray of the same resolution as the
        reference image. Since the reference image is typically a much
        higher resolution, the projected pixels will be much more sparse
        in the higher resolution space.

        PRECONDITIONS:
        * ideally, the heatmaps have been passed through an intensity high-pass
        filter

        Returns
        * dictionary containing a list of high resolution ndarrays with
          sparse pixel projections
        """
        # pudb.set_trace()
        index                       = 0
        coordInferenceSpace         = None
        coordReferenceSpace         = None
        self.lnd_heatmapHPF         = d_filteredHeatMap['images']
        for nd_heatmap in self.lnd_heatmapHPF:
            # get a list of coord pairs of nonzero heatmap pixels
            nonzeroIndices          = np.transpose(np.nonzero(nd_heatmap))
            LOG("TRANSFORMING: inference heatmap %d shape %s to reference shape %s using kernel size %d..." %
                    (index, nd_heatmap.shape, self.nd_image.shape, self.heatmapKernel))
            for coordInferenceSpace in nonzeroIndices:
                f_value             = nd_heatmap[coordInferenceSpace[0],
                                                 coordInferenceSpace[1]]
                coordInferenceSpace = np.flip(coordInferenceSpace, axis = 0)
                coordReferenceSpace = LLDcode.utils.landmark.transform.transform_coords(
                                        coordInferenceSpace,
                                        self.itk_transform
                                    )
                regionAboutCoord    = self.neighbours_findFast(2, self.heatmapKernel,
                                        np.array(coordReferenceSpace),
                                        gridSize        = self.lnd_heatmapRefSpace[index].shape,
                                        includeOrigin   = True,
                                        flipAxes        = True
                                    )
                for pixel in regionAboutCoord:
                    try:
                        # pudb.set_trace()
                        f_dist  = np.linalg.norm(pixel - np.flip(coordReferenceSpace))
                        f_scale = 1
                        self.lnd_heatmapRefSpace[index][pixel[0], pixel[1]] = f_value * f_scale
                    except:
                        LOG("\nWARNING: Transform error for heatmap about landmark [%d]" % index)
                        LOG("Inference: coordInferenceSpace  %s" % coordInferenceSpace)
                        LOG("Reference: coordRerferenceSpace %s" % coordReferenceSpace)
                        LOG("ignoring...")
            index += 1
        return {
            'heatmapsReferenceSpace'    : self.lnd_heatmapRefSpace
        }

    def heatmapList_interpolateReferenceImageSpace(self, d_referenceSpace) -> dict:
        """
        Interpolate the sparsely transformed points from the low resolution
        inference space.

        Since the aspect ratio of the reference and inference spaces differ,
        we downsample the reference space heatmap to a smaller image in a
        space close to the original heatmap -- this has the effect of
        "densifying" the sparse reference image. Then, we upsample the image
        again back to the reference space with interpolation.

        NOTE:
            The interpolation method makes a difference!
            'cv2.INTER_CUBIC' is interesting as it effectively creates a border
            artifact about the heatmap which might be academically interesting.
        """
        l_f             : list      = []
        f_aspectRatio   : float     = 0.0
        f_sigma         : float     = 2
        index           : int       = 0
        rowsRef         : int       = 0
        colsRef         : int       = 0
        rowsInf         : int       = 0
        colsInf         : int       = 0
        l_maxIntensity  : list      = []

        rowsRef, colsRef    = self.nd_image.shape
        f_aspectRatio       = colsRef / rowsRef
        rowsInf, colsInf    = self.lnd_heatmap[0].shape
        nd_downsample       = np.zeros([rowsInf, int(rowsInf * f_aspectRatio)])
        for nd_heatmapInReferenceSpace in d_referenceSpace['heatmapsReferenceSpace']:
            LOG("INTERPOLATING: Downsampling heatmap %d shape %s to shape (%d, %d)... " %\
                (index, nd_heatmapInReferenceSpace.shape, rowsInf, int(rowsInf*f_aspectRatio)),
                end="")
            # First we downsample the reference heatmap to a space close to the
            # inference space with correct aspect ration to "densify" the image...
            nd_downsample   = cv2.resize( nd_heatmapInReferenceSpace,
                                        dsize           = (int(rowsInf*f_aspectRatio), rowsInf),
                                        interpolation   = cv2.INTER_LINEAR)
            l_maxIntensity.append(np.max(nd_heatmapInReferenceSpace))
            LOG("Upsampling heatmap %d to shape (%d, %d) (linear interpolation)" % (index, rowsRef, colsRef))
            # And now we upsample again to hopefully get a nice interpolated map
            # in the larger image space
            nd_upsample     = cv2.resize(nd_downsample,
                                         dsize          = (colsRef, rowsRef),
                                         interpolation  = cv2.INTER_LINEAR)
            index       += 1
            l_f.append(nd_upsample)

        return {
            'heatmapsReferenceSpace': l_f,
            'maxIntensity'          : l_maxIntensity
        }

    def heatmapList_combine(self, d_heatmaps : dict) -> dict:
        """
        Combine (collapse) all the reference heatmap images into a single image.
        """
        LOG("COMBINING: Placing all referenceSpace heatmaps into common space...")
        nd_heatmap0         = d_heatmaps['heatmapsReferenceSpace'][0]
        nd_imageSum         = np.copy(nd_heatmap0)
        for idx in range(1, len(d_heatmaps['heatmapsReferenceSpace'])):
            nd_imageSum    += d_heatmaps['heatmapsReferenceSpace'][idx]
        return {
            'heatmapsReferenceSpace'    : d_heatmaps['heatmapsReferenceSpace'],
            'heatmapsReferenceSpaceAll' : nd_imageSum
        }

    def norm(self, nd_arr : np.array , range : int = 1, ntype = np.uint8) -> np.array:
        """
        Normalize the input <nd_arr>, optionally scaling values to range <range>
        """
        nd_norm     = nd_arr / np.max(nd_arr)
        if range > 1:
            nd_norm = (nd_norm * range).astype(ntype)
        return nd_norm

    def images_combine(self,
                        np_im1 : np.array,
                        np_im2 : np.array,
                        weight : tuple = (0.5, 0.5) ) -> np.array:
        """
        Create a composite image between two images, with per-image
        weighting as specified.
        """
        LOG("COMPOSITING: Placing reference heatmaps over reference image with weight %s" %\
            str(weight))
        np_im1n                 = self.norm(np_im1) * weight[0]
        np_im2n                 = self.norm(np_im2) * weight[1]

        return self.norm(np_im1n + np_im2n)

    def heatmapList_onInput(self, d_heatmaps : dict) -> dict:
        """
        Create a composite image with the heatmaps in the same image as the
        input -- this is rather naive matrix addition.
        """
        nd_heatmapsOnInput          = self.images_combine(
                                            d_heatmaps['heatmapsReferenceSpaceAll'],
                                            self.nd_image,
                                            self.compositeWeight
                                    )*256
        return {
            'heatmapsReferenceSpace'    : d_heatmaps['heatmapsReferenceSpace'],
            'heatmapsReferenceSpaceAll' : d_heatmaps['heatmapsReferenceSpaceAll'],
            'heatmapsOnInput'           : nd_heatmapsOnInput
        }

    def dir_checkAndCreate(self, path_dir : Path):
        """
        Check on a pathlib dir argument and create/chmod if needed.
        """
        if not path_dir.is_dir():
            path_dir.mkdir(parents = True, exist_ok = True)
            path_dir.chmod(0o777)

    def save(self, d_heatmaps : dict, imtype : str = 'jpg'):
        """
        Save resultant images:

            referenceSpace: self.path_outputDir/referenceSpace
            inferenceSpace: self.path_outputDir/inferenceSpace

        The inferenceSpace additionally contains 'original' and
        'highPassFiltered' for the original and intensity filtered
        heatmaps.

        """

        str_refImageStem    : str   = 'reference'   # subdir for reference space
        str_heatImageStem   : str   = 'heatmap'     # stem name for heatmaps
        path_reference      : Path  = None          # path to reference space
        path_inference      : Path  = None          # path to inference space
        heatMapCount        : int   = 0             # index of current heatmap
        str_fileName        : str   = ""            # rolling heatmap filename
        str_referenceImg    : str   = ""            # heatmap name in reference
        str_inferenceImg    : str   = ""            # heatmap name in inference
        str_inferenceHPF    : str   = ""            # HPF heatmap name in inference

        def dir_checkAndCreate():
            nonlocal path_reference, path_inference
            path_reference      = self.path_outputDir / 'referenceSpace'
            path_inference      = self.path_outputDir / 'inferenceSpace'
            self.dir_checkAndCreate(path_reference)
            self.dir_checkAndCreate(path_inference / 'original')
            self.dir_checkAndCreate(path_inference / 'highPassFiltered')

        # pudb.set_trace()
        dir_checkAndCreate()
        for heatmapRef, heatmapInf, heatmapInfHPF in zip(
                d_heatmaps['heatmapsReferenceSpace'],
                self.lnd_heatmap,
                self.lnd_heatmapHPF):
            str_fileName        = '%s%02d.%s' % (str_heatImageStem, heatMapCount, imtype)
            str_referenceImg    = str(path_reference / str_fileName)
            str_inferenceImg    = str(path_inference / 'original'         / str_fileName)
            str_inferenceHPF    = str(path_inference / 'highPassFiltered' / str_fileName)
            heatMapCount       += 1
            LOG("Saving normalized heatHPF in reference space %s..." % str_referenceImg)
            imageio.imwrite(str_referenceImg, self.norm(heatmapRef, 255))
            LOG("Saving normalized heatmap in inference space %s..." % str_inferenceImg)
            imageio.imwrite(str_inferenceImg, self.norm(heatmapInf, 255))
            LOG("Saving normalized heatHPF in inference space %s..." % str_inferenceHPF)
            imageio.imwrite(str_inferenceHPF, self.norm(heatmapInfHPF, 255))

        str_combinedHeatmap     = 'heatmapAll.%s'    % imtype
        LOG("Saving combined heatmaps  in reference space %s..." % \
            str(path_reference / str_combinedHeatmap))
        imageio.imwrite(str(path_reference / str_combinedHeatmap),
                        self.norm(d_heatmaps['heatmapsReferenceSpaceAll'], 255))

        str_heatmapsOnInput     = 'inputWithHeatmaps.%s'    % imtype
        LOG("Saving combined heatmaps  on input reference %s..." % \
            str(path_reference / str_heatmapsOnInput))
        imageio.imwrite(str(path_reference / str_heatmapsOnInput),
                        d_heatmaps['heatmapsOnInput'].astype(np.uint8))

        str_inputImageName      = 'input.%s'    % imtype
        LOG("Saving original   input   in reference space %s..." % str(path_reference / str_inputImageName))
        imageio.imwrite(str(path_reference / str_inputImageName), self.nd_image.astype(np.uint8))

    def run(self):
        """
        'run' this class
        """
        # pudb.set_trace()
        self.save(
            self.heatmapList_onInput(
                self.heatmapList_combine(
                    self.heatmapList_interpolateReferenceImageSpace(
                        self.heatmaplist_transformToReferenceImageSpace(
                            self.imagelist_intensityFilter(
                                    self.lnd_heatmap,
                                    filter              = 'highpass',
                                    thesholdPerc        = self.heatmapFilter)
                        )
                    )
                )
            ),
            self.imageType
        )
