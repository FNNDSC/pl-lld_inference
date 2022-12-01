
import os
import tensorflow as tf
import numpy as np
import sys
from LLDcode.tensorflow_train.utils.summary_handler import SummaryHandler, create_summary_placeholder
from LLDcode.utils.io.common import create_directories, copy_files_to_folder
import datetime
from collections import OrderedDict
from glob import glob

class MainLoopBase(object):
    def __init__(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.coord = tf.train.Coordinator()
        self.first_iteration = True
        self.train_queue = None
        self.val_queue = None
        self.batch_size = None
        self.learning_rate = None
        self.optimizer = None
        self.optimization_function = None
        self.current_iter = 0
        self.disp_iter = 1
        self.layer_weight_summary_iter = None
        self.layer_weight_inspector = None
        self.max_iter = None
        self.snapshot_iter = None
        self.test_iter = None
        self.test_initialization = True
        self.train_losses = None
        self.val_losses = None
        self.is_closed = False
        self.output_folder = ''
        self.load_model_filename = None
        self.files_to_copy = []
        self.additional_summaries_placeholders_val = None
        self.raise_on_nan_loss = True
        self.loss_name_for_nan_loss_check = 'loss'

    def init_saver(self):
        # initialize variables
        self.saver = tf.train.Saver(max_to_keep=1000)

    def init_variables(self):
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())

    def load_model(self):
        model_filename = "/usr/local/lib/lld/model-20000"
        print('Restoring model ' + model_filename)
        self.restore_variables(self.sess, model_filename)

    def restore_variables(self, session, model_filename):
        self.saver.restore(session, model_filename)

    def create_output_folder(self):
        create_directories(self.output_folder)
        if self.files_to_copy is not None:
            all_files_to_copy = []
            for file_to_copy in self.files_to_copy:
                all_files_to_copy += glob(file_to_copy)
            copy_files_to_folder(all_files_to_copy, self.output_folder)

    def output_folder_timestamp(self):
        return datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    def output_folder_for_current_iteration(self):
        return os.path.join(self.output_folder, 'results')

    def output_file_for_current_iteration(self, *args):
        return os.path.join(self.output_folder, 'results', *args)

    def init_all(self):
        self.init_networks()
        self.initLossAggregators()
        self.init_variables()
        self.init_saver()
        self.create_output_folder()

    def stop_threads(self):
        self.coord.request_stop()
        if self.train_queue is not None:
            self.train_queue.close(self.sess)
        if self.val_queue is not None:
            self.val_queue.close(self.sess)
        self.coord.join(self.threads)

    def process(self):
        self.init_all()
        self.load_model()
        print('Starting main processing loop')
        self.inference_do()

    def initLossAggregators(self):

        summaries_placeholders = OrderedDict([(loss_name, create_summary_placeholder(loss_name)) for loss_name in self.train_losses.keys()])

        summaries_placeholders_val = summaries_placeholders.copy()

        if self.additional_summaries_placeholders_val is not None:
            summaries_placeholders_val.update(self.additional_summaries_placeholders_val)

        self.val_loss_aggregator = SummaryHandler(self.sess,
                                                  self.val_losses,
                                                  summaries_placeholders_val,
                                                  'events',
                                                  os.path.join(self.output_folder, 'events'),
                                                  os.path.join(self.output_folder, 'log.csv'))


    def init_networks(self):
        print('initNetworks() is deprecated and may be removed in later versions. Use init_networks() instead.')
        self.initNetworks()

    def initNetworks(self):
        raise NotImplementedError()

    def test(self):
        raise NotImplementedError()

    def close(self):
        if not self.is_closed:
            self.stop_threads()
            self.sess.close()
            tf.reset_default_graph()
            self.is_closed = True
