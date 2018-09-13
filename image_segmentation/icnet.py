import logging
import os

from keras.layers import Activation
from keras.layers import Conv2D
from keras.layers import Add
from keras.layers import MaxPooling2D
from keras.layers import AveragePooling2D
from keras.layers import ZeroPadding2D
from keras.layers import Input
from keras.layers import BatchNormalization
from keras.layers import UpSampling2D
from keras.models import Model

from tensorflow.python.lib.io import file_io


logger = logging.getLogger('icnet')


class ICNetModelFactory(object):
    """Generates ICNet Keras Models."""

    @staticmethod
    def build(
            img_size,
            n_classes,
            alpha=1.0,
            weights_path=None,
            train=False,
            input_tensor=None):
        """Build an ICNet Model.

        Args:image_size (int): the size of each image. only square images are
                supported.
            n_classes (int): the number of output labels to predict.
            weights_path (str): (optional) a path to a Keras model file to
                load after the network is constructed. Useful for re-training.
            train (bool): (optional) if true, add additional output nodes to
                the network for training.

        Returns:
            model (keras.models.Model): A Keras model
        """
        if img_size % 384 != 0:
            raise Exception('`img_size` must be a multiple of 384.')
        logger.info('Building ICNet model.')
        inpt = Input(shape=(img_size, img_size, 3), tensor=input_tensor)

        # The 1/2 scale branch
        out_a = AveragePooling2D(pool_size=(2, 2), name='data_sub2')(inpt)
        out_a = Conv2D(int(alpha * 32), 3, strides=2, padding='same', activation='relu',
                       name='conv1_1_3x3_s2')(out_a)
        out_a = BatchNormalization(name='conv1_1_3x3_s2_bn')(out_a)
        out_a = Conv2D(int(alpha * 32), 3, padding='same', activation='relu',
                       name='conv1_2_3x3')(out_a)
        out_a = BatchNormalization(name='conv1_2_3x3_s2_bn')(out_a)
        out_a = Conv2D(int(alpha * 64), 3, padding='same', activation='relu',
                       name='conv1_3_3x3')(out_a)
        out_a = BatchNormalization(name='conv1_3_3x3_bn')(out_a)
        out_b = MaxPooling2D(pool_size=3, strides=2,
                             name='pool1_3x3_s2')(out_a)
        out_a = Conv2D(int(alpha * 128), 1, name='conv2_1_1x1_proj')(out_b)
        out_a = BatchNormalization(name='conv2_1_1x1_proj_bn')(out_a)
        out_b = Conv2D(int(alpha * 32), 1, activation='relu',
                       name='conv2_1_1x1_reduce')(out_b)
        out_b = BatchNormalization(name='conv2_1_1x1_reduce_bn')(out_b)
        out_b = ZeroPadding2D(name='padding1')(out_b)
        out_b = Conv2D(int(alpha * 32), 3, activation='relu', name='conv2_1_3x3')(out_b)
        out_b = BatchNormalization(name='conv2_1_3x3_bn')(out_b)
        out_b = Conv2D(int(alpha * 128), 1, name='conv2_1_1x1_increase')(out_b)
        out_b = BatchNormalization(name='conv2_1_1x1_increase_bn')(out_b)
        out_a = Add(name='conv2_1')([out_a, out_b])
        out_b = Activation('relu', name='conv2_1/relu')(out_a)

        out_a = Conv2D(int(alpha * 32), 1, activation='relu',
                       name='conv2_2_1x1_reduce')(out_b)
        out_a = BatchNormalization(name='conv2_2_1x1_reduce_bn')(out_a)
        out_a = ZeroPadding2D(name='padding2')(out_a)
        out_a = Conv2D(int(alpha * 32), 3, activation='relu', name='conv2_2_3x3')(out_a)
        out_a = BatchNormalization(name='conv2_2_3x3_bn')(out_a)
        out_a = Conv2D(int(alpha * 128), 1, name='conv2_2_1x1_increase')(out_a)
        out_a = BatchNormalization(name='conv2_2_1x1_increase_bn')(out_a)
        out_a = Add(name='conv2_2')([out_a, out_b])
        out_b = Activation('relu', name='conv2_2/relu')(out_a)

        out_a = Conv2D(int(alpha * 32), 1, activation='relu',
                       name='conv2_3_1x1_reduce')(out_b)
        out_a = BatchNormalization(name='conv2_3_1x1_reduce_bn')(out_a)
        out_a = ZeroPadding2D(name='padding3')(out_a)
        out_a = Conv2D(int(alpha * 32), 3, activation='relu', name='conv2_3_3x3')(out_a)
        out_a = BatchNormalization(name='conv2_3_3x3_bn')(out_a)
        out_a = Conv2D(int(alpha * 128), 1, name='conv2_3_1x1_increase')(out_a)
        out_a = BatchNormalization(name='conv2_3_1x1_increase_bn')(out_a)
        out_a = Add(name='conv2_3')([out_a, out_b])
        out_b = Activation('relu', name='conv2_3/relu')(out_a)

        out_a = Conv2D(int(alpha * 256), 1, strides=2, name='conv3_1_1x1_proj')(out_b)
        out_a = BatchNormalization(name='conv3_1_1x1_proj_bn')(out_a)
        out_b = Conv2D(int(alpha * 64), 1, strides=2, activation='relu',
                       name='conv3_1_1x1_reduce')(out_b)
        out_b = BatchNormalization(name='conv3_1_1x1_reduce_bn')(out_b)
        out_b = ZeroPadding2D(name='padding4')(out_b)
        out_b = Conv2D(int(alpha * 64), 3, activation='relu', name='conv3_1_3x3')(out_b)
        out_b = BatchNormalization(name='conv3_1_3x3_bn')(out_b)
        out_b = Conv2D(int(alpha * 256), 1, name='conv3_1_1x1_increase')(out_b)
        out_b = BatchNormalization(name='conv3_1_1x1_increase_bn')(out_b)
        out_a = Add(name='conv3_1')([out_a, out_b])
        out_c = Activation('relu', name='conv3_1/relu')(out_a)

        # The 1/4 scale branch
        out_b = AveragePooling2D(pool_size=(2, 2), name='conv3_1_sub4')(out_c)
        out_a = Conv2D(int(alpha * 64), 1, activation='relu',
                       name='conv3_2_1x1_reduce')(out_b)
        out_a = BatchNormalization(name='conv3_2_1x1_reduce_bn')(out_a)
        out_a = ZeroPadding2D(name='padding5')(out_a)
        out_a = Conv2D(int(alpha * 64), 3, activation='relu', name='conv3_2_3x3')(out_a)
        out_a = BatchNormalization(name='conv3_2_3x3_bn')(out_a)
        out_a = Conv2D(int(alpha * 256), 1, name='conv3_2_1x1_increase')(out_a)
        out_a = BatchNormalization(name='conv3_2_1x1_increase_bn')(out_a)
        out_a = Add(name='conv3_2')([out_a, out_b])
        out_b = Activation('relu', name='conv3_2/relu')(out_a)

        out_a = Conv2D(int(alpha * 64), 1, activation='relu',
                       name='conv3_3_1x1_reduce')(out_b)
        out_a = BatchNormalization(name='conv3_3_1x1_reduce_bn')(out_a)
        out_a = ZeroPadding2D(name='padding6')(out_a)
        out_a = Conv2D(int(alpha * 64), 3, activation='relu', name='conv3_3_3x3')(out_a)
        out_a = BatchNormalization(name='conv3_3_3x3_bn')(out_a)
        out_a = Conv2D(int(alpha * 256), 1, name='conv3_3_1x1_increase')(out_a)
        out_a = BatchNormalization(name='conv3_3_1x1_increase_bn')(out_a)
        out_a = Add(name='conv3_3')([out_a, out_b])
        out_b = Activation('relu', name='conv3_3/relu')(out_a)

        out_a = Conv2D(int(alpha * 64), 1, activation='relu',
                       name='conv3_4_1x1_reduce')(out_b)
        out_a = BatchNormalization(name='conv3_4_1x1_reduce_bn')(out_a)
        out_a = ZeroPadding2D(name='padding7')(out_a)
        out_a = Conv2D(int(alpha * 64), 3, activation='relu', name='conv3_4_3x3')(out_a)
        out_a = BatchNormalization(name='conv3_4_3x3_bn')(out_a)
        out_a = Conv2D(int(alpha * 256), 1, name='conv3_4_1x1_increase')(out_a)
        out_a = BatchNormalization(name='conv3_4_1x1_increase_bn')(out_a)
        out_a = Add(name='conv3_4')([out_a, out_b])
        out_b = Activation('relu', name='conv3_4/relu')(out_a)

        out_a = Conv2D(int(alpha * 512), 1, name='conv4_1_1x1_proj')(out_b)
        out_a = BatchNormalization(name='conv4_1_1x1_proj_bn')(out_a)
        out_b = Conv2D(int(alpha * 128), 1, activation='relu',
                       name='conv4_1_1x1_reduce')(out_b)
        out_b = BatchNormalization(name='conv4_1_1x1_reduce_bn')(out_b)
        out_b = ZeroPadding2D(padding=2, name='padding8')(out_b)
        out_b = Conv2D(int(alpha * 128), 3, dilation_rate=2, activation='relu',
                       name='conv4_1_3x3')(out_b)
        out_b = BatchNormalization(name='conv4_1_3x3_bn')(out_b)
        out_b = Conv2D(int(alpha * 512), 1, name='conv4_1_1x1_increase')(out_b)
        out_b = BatchNormalization(name='conv4_1_1x1_increase_bn')(out_b)
        out_a = Add(name='conv4_1')([out_a, out_b])
        out_b = Activation('relu', name='conv4_1/relu')(out_a)

        out_a = Conv2D(int(alpha * 128), 1, activation='relu',
                       name='conv4_2_1x1_reduce')(out_b)
        out_a = BatchNormalization(name='conv4_2_1x1_reduce_bn')(out_a)
        out_a = ZeroPadding2D(padding=2, name='padding9')(out_a)
        out_a = Conv2D(int(alpha * 128), 3, dilation_rate=2, activation='relu',
                       name='conv4_2_3x3')(out_a)
        out_a = BatchNormalization(name='conv4_2_3x3_bn')(out_a)
        out_a = Conv2D(int(alpha * 512), 1, name='conv4_2_1x1_increase')(out_a)
        out_a = BatchNormalization(name='conv4_2_1x1_increase_bn')(out_a)
        out_a = Add(name='conv4_2')([out_a, out_b])
        out_b = Activation('relu', name='conv4_2/relu')(out_a)

        out_a = Conv2D(int(alpha * 128), 1, activation='relu',
                       name='conv4_3_1x1_reduce')(out_b)
        out_a = BatchNormalization(name='conv4_3_1x1_reduce_bn')(out_a)
        out_a = ZeroPadding2D(padding=2, name='padding10')(out_a)
        out_a = Conv2D(int(alpha * 128), 3, dilation_rate=2, activation='relu',
                       name='conv4_3_3x3')(out_a)
        out_a = BatchNormalization(name='conv4_3_3x3_bn')(out_a)
        out_a = Conv2D(int(alpha * 512), 1, name='conv4_3_1x1_increase')(out_a)
        out_a = BatchNormalization(name='conv4_3_1x1_increase_bn')(out_a)
        out_a = Add(name='conv4_3')([out_a, out_b])
        out_b = Activation('relu', name='conv4_3/relu')(out_a)

        out_a = Conv2D(int(alpha * 128), 1, activation='relu',
                       name='conv4_4_1x1_reduce')(out_b)
        out_a = BatchNormalization(name='conv4_4_1x1_reduce_bn')(out_a)
        out_a = ZeroPadding2D(padding=2, name='padding11')(out_a)
        out_a = Conv2D(int(alpha * 128), 3, dilation_rate=2, activation='relu',
                       name='conv4_4_3x3')(out_a)
        out_a = BatchNormalization(name='conv4_4_3x3_bn')(out_a)
        out_a = Conv2D(int(alpha * 512), 1, name='conv4_4_1x1_increase')(out_a)
        out_a = BatchNormalization(name='conv4_4_1x1_increase_bn')(out_a)
        out_a = Add(name='conv4_4')([out_a, out_b])
        out_b = Activation('relu', name='conv4_4/relu')(out_a)

        out_a = Conv2D(int(alpha * 128), 1, activation='relu',
                       name='conv4_5_1x1_reduce')(out_b)
        out_a = BatchNormalization(name='conv4_5_1x1_reduce_bn')(out_a)
        out_a = ZeroPadding2D(padding=2, name='padding12')(out_a)
        out_a = Conv2D(int(alpha * 128), 3, dilation_rate=2, activation='relu',
                       name='conv4_5_3x3')(out_a)
        out_a = BatchNormalization(name='conv4_5_3x3_bn')(out_a)
        out_a = Conv2D(int(alpha * 512), 1, name='conv4_5_1x1_increase')(out_a)
        out_a = BatchNormalization(name='conv4_5_1x1_increase_bn')(out_a)
        out_a = Add(name='conv4_5')([out_a, out_b])
        out_b = Activation('relu', name='conv4_5/relu')(out_a)

        out_a = Conv2D(int(alpha * 128), 1, activation='relu',
                       name='conv4_6_1x1_reduce')(out_b)
        out_a = BatchNormalization(name='conv4_6_1x1_reduce_bn')(out_a)
        out_a = ZeroPadding2D(padding=2, name='padding13')(out_a)
        out_a = Conv2D(int(alpha * 128), 3, dilation_rate=2, activation='relu',
                       name='conv4_6_3x3')(out_a)
        out_a = BatchNormalization(name='conv4_6_3x3_bn')(out_a)
        out_a = Conv2D(int(alpha * 512), 1, name='conv4_6_1x1_increase')(out_a)
        out_a = BatchNormalization(name='conv4_6_1x1_increase_bn')(out_a)
        out_a = Add(name='conv4_6')([out_a, out_b])
        out_a = Activation('relu', name='conv4_6/relu')(out_a)

        out_b = Conv2D(int(alpha * 1024), 1, name='conv5_1_1x1_proj')(out_a)
        out_b = BatchNormalization(name='conv5_1_1x1_proj_bn')(out_b)
        out_a = Conv2D(int(alpha * 256), 1, activation='relu',
                       name='conv5_1_1x1_reduce')(out_a)
        out_a = BatchNormalization(name='conv5_1_1x1_reduce_bn')(out_a)
        out_a = ZeroPadding2D(padding=4, name='padding14')(out_a)
        out_a = Conv2D(int(alpha * 256), 3, dilation_rate=4, activation='relu',
                       name='conv5_1_3x3')(out_a)
        out_a = BatchNormalization(name='conv5_1_3x3_bn')(out_a)
        out_a = Conv2D(int(alpha * 1024), 1, name='conv5_1_1x1_increase')(out_a)
        out_a = BatchNormalization(name='conv5_1_1x1_increase_bn')(out_a)
        out_a = Add(name='conv5_1')([out_a, out_b])
        out_b = Activation('relu', name='conv5_1/relu')(out_a)

        out_a = Conv2D(int(alpha * 256), 1, activation='relu',
                       name='conv5_2_1x1_reduce')(out_b)
        out_a = BatchNormalization(name='conv5_2_1x1_reduce_bn')(out_a)
        out_a = ZeroPadding2D(padding=4, name='padding15')(out_a)
        out_a = Conv2D(int(alpha * 256), 3, dilation_rate=4, activation='relu',
                       name='conv5_2_3x3')(out_a)
        out_a = BatchNormalization(name='conv5_2_3x3_bn')(out_a)
        out_a = Conv2D(int(alpha * 1024), 1, name='conv5_2_1x1_increase')(out_a)
        out_a = BatchNormalization(name='conv5_2_1x1_increase_bn')(out_a)
        out_a = Add(name='conv5_2')([out_a, out_b])
        out_b = Activation('relu', name='conv5_2/relu')(out_a)

        out_a = Conv2D(int(alpha * 256), 1, activation='relu',
                       name='conv5_3_1x1_reduce')(out_b)
        out_a = BatchNormalization(name='conv5_3_1x1_reduce_bn')(out_a)
        out_a = ZeroPadding2D(padding=4, name='padding16')(out_a)
        out_a = Conv2D(int(alpha * 256), 3, dilation_rate=4, activation='relu',
                       name='conv5_3_3x3')(out_a)
        out_a = BatchNormalization(name='conv5_3_3x3_bn')(out_a)
        out_a = Conv2D(int(alpha * 1024), 1, name='conv5_3_1x1_increase')(out_a)
        out_a = BatchNormalization(name='conv5_3_1x1_increase_bn')(out_a)
        out_a = Add(name='conv5_3')([out_a, out_b])
        out_a = Activation('relu', name='conv5_3/relu')(out_a)

        # In this version we've fixed the input dimensions to be square
        # We also are restricting dimsensions to be multiples of 384 which
        # will allow us to use standard upsampling layers for resizing.
        pool_height, _ = out_a.shape[1:3].as_list()
        pool_scale = int(img_size / 384)
        pool1 = AveragePooling2D(
            pool_size=pool_height,
            strides=pool_height,
            name='conv5_3_pool1'
        )(out_a)
        pool1 = UpSampling2D(size=12 * pool_scale,
                             name='conv5_3_pool1_interp')(pool1)
        pool2 = AveragePooling2D(pool_size=pool_height // 2,
                                 strides=pool_height // 2,
                                 name='conv5_3_pool2')(out_a)
        pool2 = UpSampling2D(size=6 * pool_scale,
                             name='conv5_3_pool2_interp')(pool2)
        pool3 = AveragePooling2D(pool_size=pool_height // 3,
                                 strides=pool_height // 3,
                                 name='conv5_3_pool3')(out_a)
        pool3 = UpSampling2D(size=4 * pool_scale,
                             name='conv5_3_pool3_interp')(pool3)
        pool4 = AveragePooling2D(pool_size=pool_height // 4,
                                 strides=pool_height // 4,
                                 name='conv5_3_pool4')(out_a)
        pool4 = UpSampling2D(size=3 * pool_scale,
                             name='conv5_3_pool6_interp')(pool4)

        out_a = Add(name='conv5_3_sum')([out_a, pool1, pool2, pool3, pool4])
        out_a = Conv2D(int(alpha * 256), 1, activation='relu', name='conv5_4_k1')(out_a)
        out_a = BatchNormalization(name='conv5_4_k1_bn')(out_a)
        aux_1 = UpSampling2D(size=(2, 2), name='conv5_4_interp')(out_a)
        out_a = ZeroPadding2D(padding=2, name='padding17')(aux_1)
        out_a = Conv2D(int(alpha * 128), 3, dilation_rate=2, name='conv_sub4')(out_a)
        out_a = BatchNormalization(name='conv_sub4_bn')(out_a)
        out_b = Conv2D(int(alpha * 128), 1, name='conv3_1_sub2_proj')(out_c)
        out_b = BatchNormalization(name='conv3_1_sub2_proj_bn')(out_b)
        out_a = Add(name='sub24_sum')([out_a, out_b])
        out_a = Activation('relu', name='sub24_sum/relu')(out_a)

        aux_2 = UpSampling2D(size=(2, 2), name='sub24_sum_interp')(out_a)
        out_a = ZeroPadding2D(padding=2, name='padding18')(aux_2)
        out_b = Conv2D(int(alpha * 128), 3, dilation_rate=2, name='conv_sub2')(out_a)
        out_b = BatchNormalization(name='conv_sub2_bn')(out_b)

        # The full scale branch
        out_a = Conv2D(int(alpha * 32), 3, strides=2, padding='same', activation='relu',
                       name='conv1_sub1')(inpt)
        out_a = BatchNormalization(name='conv1_sub1_bn')(out_a)
        out_a = Conv2D(int(alpha * 32), 3, strides=2, padding='same', activation='relu',
                       name='conv2_sub1')(out_a)
        out_a = BatchNormalization(name='conv2_sub1_bn')(out_a)
        out_a = Conv2D(int(alpha * 64), 3, strides=2, padding='same', activation='relu',
                       name='conv3_sub1')(out_a)
        out_a = BatchNormalization(name='conv3_sub1_bn')(out_a)
        out_a = Conv2D(int(alpha * 128), 1, name='conv3_sub1_proj')(out_a)
        out_a = BatchNormalization(name='conv3_sub1_proj_bn')(out_a)

        out_a = Add(name='sub12_sum')([out_a, out_b])
        out_a = Activation('relu', name='sub12_sum/relu')(out_a)
        out_a = UpSampling2D(size=(2, 2), name='sub12_sum_interp')(out_a)

        out = Conv2D(n_classes, 1, activation='softmax',
                     name='conv6_cls')(out_a)

        if train:
            aux_1 = Conv2D(n_classes, 1, activation='softmax',
                           name='sub4_out')(aux_1)
            aux_2 = Conv2D(n_classes, 1, activation='softmax',
                           name='sub24_out')(aux_2)
            model = Model(inputs=inpt, outputs=[out, aux_2, aux_1])
        else:
            model = Model(inputs=inpt, outputs=out)

        if weights_path is not None:
            if weights_path.startswith('gs://'):
                weights_path = _copy_file_from_gcs(weights_path)
            logger.info('Loading weights from %s.' % weights_path)
            model.load_weights(weights_path, by_name=True)
        logger.info('Done building model.')
        return model


def _copy_file_from_gcs(file_path):
    """Copy a file from gcs to local machine.

    Args:
        file_path (str): a GCS url to download
    Returns:
        str: a local path to the file
    """
    logger.info('Downloading %s' % file_path)
    with file_io.FileIO(file_path, mode='rb') as input_f:
        basename = os.path.basename(file_path)
        local_path = os.path.join('/tmp', basename)
        with file_io.FileIO(local_path, mode='w+') as output_f:
            output_f.write(input_f.read())
    return local_path
