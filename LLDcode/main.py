
import os
import numpy as np
import tensorflow as tf
from collections import OrderedDict

import LLDcode.tensorflow_train
import LLDcode.tensorflow_train.utils.tensorflow_util
from LLDcode.tensorflow_train.utils.data_format import get_batch_channel_image_size
import LLDcode.utils.io.image
import LLDcode.utils.io.landmark
import LLDcode.utils.io.text
from LLDcode.tensorflow_train.data_generator import DataGenerator
from LLDcode.tensorflow_train.train_loop import MainLoopBase
from LLDcode.utils.landmark.heatmap_test import HeatmapTest

from datetime import datetime
import time

from .dataset import Dataset
from .network import network_scn, network_unet, network_downsampling, network_conv, network_scn_mmwhs

from    PIL             import  Image as pImage
import  cv2
import  imageio
import  SimpleITK       as      sitk
from    SimpleITK       import  Image
from    pathlib         import  Path
import  pudb
import  tempfile
from    functools       import  reduce
from    operator        import  mul


class MainLoop(MainLoopBase):
    def __init__(self, cv, network_id,inputdir,outputdir):
        super().__init__()
        self.cv = cv
        self.network_id = network_id
        self.output_folder = outputdir
        self.batch_size = 8
        self.learning_rate = 0.0000001 # TODO adapt learning rates for different networks for faster training
        self.max_iter = 20000
        self.test_iter = 5000
        self.disp_iter = 10
        self.snapshot_iter = self.test_iter
        self.test_initialization = True
        self.current_iter = 0
        self.reg_constant = 0.00005
        self.invert_transformation = False
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

    def test(self):
        heatmap_test = HeatmapTest(channel_axis=0, invert_transformation=self.invert_transformation)
        landmarks = {}
        tic=time.perf_counter()
        pudb.set_trace()
        for i in range(self.dataset_val.num_entries()):
            dataset_entry = self.dataset_val.get_next()
            current_id = dataset_entry['id']
            print(f"Currently processing {current_id}")
            datasources = dataset_entry['datasources']
            reference_image = datasources['image_datasource']
            image, prediction, transform = self.test_full_image(dataset_entry)
            LLDcode.utils.io.image.write_np((prediction * 128).astype(np.int8), self.output_file_for_current_iteration(current_id + '_heatmap.mha'))
            predicted_landmarks = heatmap_test.get_landmarks(prediction, reference_image, transformation=transform)
            p2r     = p2r_transform( reference      = reference_image,
                                     landmarks      = predicted_landmarks,
                                     predictions    = prediction)
            p2r.run()
            LLDcode.tensorflow_train.utils.tensorflow_util.print_progress_bar(i, self.dataset_val.num_entries())
            landmarks[current_id] = predicted_landmarks


        toc=time.perf_counter()
        LLDcode.tensorflow_train.utils.tensorflow_util.print_progress_bar(self.dataset_val.num_entries(), self.dataset_val.num_entries())
        LLDcode.utils.io.landmark.save_points_csv(landmarks, self.output_file_for_current_iteration('prediction.csv'))
        print(f"total execute time duration = {toc-tic:4.4f} seconds")


    def run(inputdir,outputdir):
        network = 'conv'
        loop = MainLoop(0, network,inputdir,outputdir)
        loop.run_test()

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
    itself, if naively mapped back into the reference space, might not
    match the landmark location.

    """

    def __init__(self, *args, **kwargs):
        self.imageITK_reference     : Image         = None
        self.image                  : np.ndarray    = None
        self.l_heatmap              : list          = []
        self.heatmap                : np.ndarray    = None
        self.l_landmarks            : list          = []

        self.mmFilter               = sitk.MinimumMaximumImageFilter()

        # High pass filter on heatmap values (normalized)
        self.f_HPFheatmap           : float         = 0.85

        for k,v in kwargs.items():
            if k == 'reference'     : self.imageITK_reference   = v
            if k == 'landmarks'     : self.l_landmarks          = v
            if k == 'predictions'   : self.l_heatmap            = v
            if k == 'HPFheatmap'    : self.f_HPFheatmap         = v

        self.image      = sitk.GetArrayFromImage(self.imageITK_reference)
        self.imageInt   = self.image.astype(np.uint8)

    @property
    def HPFheatmap(self):
        return self.f_HPFheatmap

    @HPFheatmap.setter
    def HPFheatmap(self, f_v):
        self.f_HPFheatmap = f_v

    def heatmaps_intensityHighPassFilter(self):
        """
        Simple high pass filter on intensity values in heatmap matrices.
        This is an in-place modification of the internal self.l_heatmap list!
        """
        l_image         : list  = []
        l_filtered      : list  = []
        for heatmap in self.l_heatmap:
            l_image     = heatmap.tolist()
            l_filtered  = [[x if x > self.f_HPFheatmap else 0.0 for x in y] for y in l_image]
            heatmap     = np.array(l_filtered)

    def heatmaps_transformToReferenceImage(self):
        """
        Transform the heatmaps into a new matrix/array in the coordinate system
        (size) as the reference image.

        This re-uses existing code, lightly copy/pasted and factored into this class
        so to have minimal impact on existing code.

        PRECONDITIONS:
        * ideally, the heatmaps have been passed through an intensity high-pass filter
        """
        pass


    def landmarks_combine(self):
        """
        Combine (collapse) all the landmark images into a single image
        and reduce noise with cv2.normalize()
        """
        imageSum        = reduce(mul, self.l_heatmap)
        # imageAve        = imageSum / len(self.l_heatmap)
        imageAve        = imageSum
        self.heatmap    = cv2.normalize(imageAve, None, 0, 255, cv2.NORM_MINMAX)

    def save(self, toPath : Path, imtype : str = 'jpg'):
        """
        Save the class data structures in 'toPath'
        """

        str_refImageStem    : str   = 'reference'
        str_heatImageStem   : str   = 'heatMap'
        str_landMarkStem    : str   = 'landMark'
        heatMapCount        : int   = 0

        toPath.mkdir(mode = 0o777, exist_ok = True)
        pudb.set_trace()
        self.mmFilter.Execute(self.imageITK_reference)

        for hm in self.l_heatmap:
            heatMapCount += 1
            imageio.imwrite(str(toPath / str(str_heatImageStem+'%02d.' % heatMapCount+imtype)), hm)
        imageio.imwrite(str(toPath / str(str_refImageStem+'.'+imtype)), self.imageInt)
        imageio.imwrite(str(toPath / str(str_heatImageStem+'.'+imtype)), self.heatmap)

    def run(self):
        """
        'run' this class
        """
        self.landmarks_combine()
        self.save(Path(tempfile.mkdtemp(prefix='lld-')))
