import os
import time
import argparse
import tensorflow as tf
from network import PixelDCN

"""
This file provides configuration to build U-NET for semantic segmentation.
"""


def configure():
    # training
    flags = tf.app.flags
    flags.DEFINE_integer('max_step', 2000000, '# of step for training')
    flags.DEFINE_integer('test_interval', 100, '# of interval to test a model')
    flags.DEFINE_integer('save_interval', 100, '# of interval to save  model')
    flags.DEFINE_integer('summary_interval', 100, '# of step to save summary')
    flags.DEFINE_float('learning_rate', 5e-4, 'learning rate')
    # data
    flags.DEFINE_string('data_dir', './dataset/', 'Name of data directory')
    flags.DEFINE_string('train_data', 'training3d.h5', 'Training data')
    flags.DEFINE_string('valid_data', 'validation3d.h5', 'Validation data')
    flags.DEFINE_string('test_data', 'testing3d.h5', 'Testing data')
    flags.DEFINE_string('data_type', '3D', '2D data or 3D data')
    flags.DEFINE_integer('batch', 2, 'batch size') # change it for training and testing 
    flags.DEFINE_integer('channel', 1, 'channel size')
    flags.DEFINE_integer('depth', 8, 'depth size')
    flags.DEFINE_integer('height', 160, 'height size')
    flags.DEFINE_integer('width', 160, 'width size')
    # Debug
    flags.DEFINE_boolean("isTraining", True, "if it is training or testing")
    flags.DEFINE_string('logdir', './logdir', 'Log dir')
    flags.DEFINE_string('modeldir', './modeldir', 'Model dir')
    flags.DEFINE_string('sampledir', './samples/', 'Sample directory')
    flags.DEFINE_string('model_name', 'model', 'Model file name')
    flags.DEFINE_integer('reload_step', 1205000, 'Reload step to continue training')
    flags.DEFINE_integer('test_step',1205000, 'Test or predict model at this step')
    flags.DEFINE_integer('random_seed', int(time.time()), 'random seed')
    # network architecturenvi   smi
    flags.DEFINE_integer('network_depth', 3, 'network depth for U-Net')
    flags.DEFINE_integer('class_num', 3, 'output class number')
    flags.DEFINE_integer('start_channel_num', 32,
                         'start number of outputs for the first conv layer')
    flags.DEFINE_string(
        'conv_name', 'conv',
        'Use which conv op in decoder: conv or ipixel_cl')
    flags.DEFINE_string(
        'deconv_name', 'deconv',
        'Use which deconv op in decoder: deconv, pixel_dcl, ipixel_dcl')
    flags.DEFINE_string(
        'action', 'add',
        'Use how to combine feature maps in pixel_dcl and ipixel_dcl: concat or add')
    # fix bug of flags
    flags.FLAGS.__dict__['__parsed'] = False
    return flags.FLAGS


def main(_):
    parser = argparse.ArgumentParser()
    parser.add_argument('--option', dest='option', type=str, default='train',
                        help='actions: train, test, or predict')
    args = parser.parse_args()
    if args.option not in ['train', 'test', 'predict']:
        print('invalid option: ', args.option)
        print("Please input a option: train, test, or predict")
    else:
        model = PixelDCN(tf.Session(), configure())
        getattr(model, args.option)()


if __name__ == '__main__':
    # configure which gpu or cpu to use
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    tf.app.run()