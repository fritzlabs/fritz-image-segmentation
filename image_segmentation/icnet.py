import logging
from functools import partial
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
    def _light_cnn_block(
            out,
            filter_scale,
            block_name,
            strides=1,
            include_projection=False):

        conv_fn = partial(
            Conv2D,
            kernel_size=3,
            strides=strides,
            padding='same',
            use_bias=False,
            activation='relu'
        )
        out = conv_fn(filters=filter_scale, name='%s_1_3x3' % block_name)(out)
        out = BatchNormalization(name='%s_1_3x3_bn' % block_name)(out)
        out = conv_fn(filters=filter_scale, name='%s_2_3x3' % block_name)(out)
        out = BatchNormalization(name='%s_2_3x3_bn' % block_name)(out)
        out = conv_fn(filters=filter_scale * 2,
                      name='%s_3_3x3' % block_name)(out)
        out = BatchNormalization(name='%s_3_3x3_bn' % block_name)(out)

        if include_projection:
            out = Conv2D(
                filters=filter_scale * 4,
                kernel_size=1,
                name='%s_proj' % block_name
            )(out)
            out = BatchNormalization(name='%s_proj_bn' % block_name)(out)

        return out

    @staticmethod
    def _inner_conv_block(out, filter_scale, block_name, dilation_rate=1):
        out = Conv2D(
            filters=filter_scale,
            kernel_size=1, 
            activation='relu',
            use_bias=False, 
            name='%s_1x1_reduce' % block_name)(out)
        out = BatchNormalization(name='%s_1x1_reduce_bn' % block_name)(out)
        out = ZeroPadding2D(
            padding=dilation_rate,
            name='%s_padding' % block_name)(out)
        out = Conv2D(
            filters=filter_scale,
            kernel_size=3,
            dilation_rate=dilation_rate,
            use_bias=False,
            activation='relu',
            name='%s_3x3' % block_name)(out)
        out = BatchNormalization(name='%s_3x3_bn' % block_name)(out)
        out = Conv2D(
            filters=filter_scale * 4,
            kernel_size=1,
            use_bias=False,
            name='%s_1x1_increase' % block_name)(out)
        out = BatchNormalization(name='%s_1x1_increase_bn' % block_name)(out)
        return out

    @classmethod
    def _conv_block(
            cls,
            out,
            filter_scale,
            block_name,
            include_projection=False,
            dilation_rate=1):
        # Branch A
        if include_projection:
            out_a = Conv2D(
                filters=filter_scale * 4,
                kernel_size=1, 
                use_bias=False,
                name='%s_1x1_proj' % block_name
            )(out)
            out_a = BatchNormalization(
                name='%s_1x1_proj_bn' % block_name
            )(out_a)
        else:
            out_a = out

        # Branch B
        out_b = cls._inner_conv_block(
            out, filter_scale, block_name, dilation_rate=dilation_rate
        )

        # Combine
        out = Add(name='%s_add' % block_name)([out_a, out_b])
        out = Activation('relu', name='%s_relu' % block_name)(out)
        return out

    @staticmethod
    def _cff_block(out_a, out_b, filter_scale, block_name, include_projection=False):
        aux_1 = UpSampling2D(size=(2, 2), name='%s_interp' % block_name)(out_a)
        out_a = ZeroPadding2D(padding=2, name='%s_padding' % block_name)(aux_1)
        out_a = Conv2D(
            filters=filter_scale,
            kernel_size=3,
            dilation_rate=2,
            use_bias=False,
            name='%s_conv_3x3' % block_name
        )(out_a)
        out_a = BatchNormalization(name='%s_conv_bn' % block_name)(out_a)

        if include_projection:
            out_b = Conv2D(
                filters=filter_scale,
                kernel_size=1,
                use_biase=False,
                name='%s_proj' % block_name)(out_b)
            out_b = BatchNormalization(name='%s_proj_bn' % block_name)(out_b)
    
        out_a = Add(name='%s_sum' % block_name)([out_a, out_b])
        out_a = Activation('relu', name='%s_sum_relu' % block_name)(out_a)

        return out_a, aux_1


    @classmethod
    def build(
            cls,
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

        # The full scale branch
        out_1 = cls._light_cnn_block(
            inpt,
            filter_scale=int(alpha * 32),
            strides=2,
            include_projection=True,
            block_name='sub1_conv'
        )

        # The 1/2 scale branch
        out_2 = AveragePooling2D(pool_size=(2, 2), name='sub2_data')(inpt)
        out_2 = cls._light_cnn_block(
            out_2, filter_scale=int(alpha * 32), block_name='sub2_conv'
        )
        out_2 = MaxPooling2D(
            pool_size=3, strides=2, name='sub2_pool1_3x3'
        )(out_2)

        for layer_index in range(1, 4):
            out_2 = cls._conv_block(
                out_2, 
                filter_scale=int(alpha * 32),
                include_projection=(layer_index == 1),
                block_name='sub2_conv%d_%d' % (2, layer_index)
            )

        # The third large conv block gets split off into another branch.
        out_2 = cls._conv_block(
            out_2, 
            filter_scale=int(alpha * 64),
            include_projection=True,
            block_name='sub2_conv%d_%d' % (3, 1)
        )

        # The 1/4 scale branch
        out_4 = AveragePooling2D(pool_size=(2, 2), name='sub4_conv3_1')(out_2)

        for layer_index in range(2, 5):
            out_4 = cls._conv_block(
                out_4,
                filter_scale=int(alpha * 64),
                block_name='sub4_conv%d_%d' % (3, layer_index)
            )

        for layer_index in range(1, 7):
            out_4 = cls._conv_block(
                out_4,
                filter_scale=int(alpha * 128),
                dilation_rate=2,
                include_projection=(layer_index == 1),
                block_name='sub4_conv%d_%d' % (4, layer_index)
            )

        for sub_index in range(1, 4):
            out_4 = cls._conv_block(
                out_4,
                filter_scale=int(alpha * 256),
                dilation_rate=4,
                include_projection=(sub_index == 1),
                block_name='sub4_conv%d_%d' % (5, sub_index)
            )
        
        # In this version we've fixed the input dimensions to be square
        # We also are restricting dimsensions to be multiples of 384 which
        # will allow us to use standard upsampling layers for resizing.
        pool_height, _ = out_4.shape[1:3].as_list()
        pool_scale = int(img_size / 384)
        pool1 = AveragePooling2D(pool_size=pool_height,
                                 strides=pool_height,
                                 name='sub4_conv5_3_pool1')(out_4)
        pool1 = UpSampling2D(size=12 * pool_scale,
                             name='sub4_conv5_3_pool1_interp')(pool1)
        pool2 = AveragePooling2D(pool_size=pool_height // 2,
                                 strides=pool_height // 2,
                                 name='sub4_conv5_3_pool2')(out_4)
        pool2 = UpSampling2D(size=6 * pool_scale,
                             name='sub4_conv5_3_pool2_interp')(pool2)
        pool3 = AveragePooling2D(pool_size=pool_height // 3,
                                 strides=pool_height // 3,
                                 name='sub4_conv5_3_pool3')(out_4)
        pool3 = UpSampling2D(size=4 * pool_scale,
                             name='sub4_conv5_3_pool3_interp')(pool3)
        pool4 = AveragePooling2D(pool_size=pool_height // 4,
                                 strides=pool_height // 4,
                                 name='sub4_conv5_3_pool4')(out_4)
        pool4 = UpSampling2D(size=3 * pool_scale,
                             name='sub4_conv5_3_pool6_interp')(pool4)
        print([l.shape.as_list() for l in [out_4, pool1, pool2, pool3, pool4]])
        out_4 = Add(
            name='sub4_conv5_3_sum'
        )([out_4, pool1, pool2, pool3, pool4])
        out_4 = Conv2D(
            filters=int(alpha * 256),
            kernel_size=1,
            activation='relu',
            use_bias=False,
            name='sub4_conv5_4_k1')(out_4)
        out_4 = BatchNormalization(name='sub4_conv5_4_k1_bn')(out_4)

        out_2, aux_1 = cls._cff_block(
            out_4,
            out_2,
            int(alpha * 128),
            block_name='sub24_cff',
            include_projection=True
        )

        out_1, aux_2 = cls._cff_block(
            out_2,
            out_1,
            int(alpha * 128),
            block_name='sub12_cff'
        )
        out_1 = UpSampling2D(size=(2, 2), name='sub12_sum_interp')(out_1)

        out = Conv2D(n_classes, 1, activation='softmax',
                     name='conv6_cls')(out_1)

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
