"""Summary.

Attributes:
    logger (TYPE): Description
"""
import logging

import numpy
import tensorflow as tf
from tensorflow.python.lib.io import file_io

logger = logging.getLogger('data_generator')


def _gaussian_kernel_3d(sigma, channels=3, size=4.0):
    radius = sigma * size / 2.0 + 0.5
    gauss = tf.distributions.Normal(0., sigma)
    kernel_1d = gauss.prob(
        tf.range(-radius[0], radius[0] + 1.0, dtype=tf.float32)
    )
    kernel_2d = tf.sqrt(tf.einsum('i,j->ij', kernel_1d, kernel_1d))
    kernel_2d = kernel_2d / tf.reduce_sum(kernel_2d)
    kernel = tf.expand_dims(kernel_2d, -1)
    kernel = tf.expand_dims(kernel, -1)
    kernel = tf.tile(kernel, [1, 1, channels, 1])
    return kernel


class ADE20KDatasetBuilder(object):
    """Create a TFRecord dataset from the ADE20K data."""

    # Scale and bias parameters to pre-process images so pixel values are
    # between -0.5 and 0.5
    _PREPROCESS_IMAGE_SCALE = 1.0 / 255.0
    _PREPROCESS_CHANNEL_BIAS = -0.5

    @staticmethod
    def load_class_labels(label_filename):
        """Load class labels.

        Assumes the data directory is left unchanged from the original zip.

        Args:
            root_directory (str): the dataset's root directory

        Returns:
            arr: an array of class labels
        """
        class_labels = []
        header = True
        with file_io.FileIO(label_filename, mode='r') as file:
            for line in file.readlines():
                if header:
                    header = False
                    continue
                line = line.rstrip()
                label = line.split('\t')[-1]
                class_labels.append(label)
        return numpy.array(class_labels)

    @staticmethod
    def _resize_fn(images, image_size):
        """Resize an input images..

        Args:
            images (tf.tensor): a tensor of input images
            image_size ((int, int)): a size (H,W) to resize to

        Returns:
            tf.tensor: a resized image tensor
        """
        return tf.image.resize_images(
            images,
            image_size,
            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
        )

    @classmethod
    def _preprocess_example(cls, example):
        """Preprocess an image.

        Args:
            example (dict): a single example from the dataset

        Return:
            (dict) processed example from the dataset
        """
        example['image'] = (tf.cast(example['image'], tf.float32) *
                            cls._PREPROCESS_IMAGE_SCALE +
                            cls._PREPROCESS_CHANNEL_BIAS)
        return example

    @classmethod
    def _resize_example(cls, example, image_size):
        """Resize an image and mask from.

        Args:
            example (dict): a single example from the dataset.
            image_size ((int, int)): the desired size of image and mask

        Returns:
            (dict) a single example resized
        """
        return {'image': cls._resize_fn(example['image'], image_size),
                'mask': cls._resize_fn(example['mask'], image_size)}

    @staticmethod
    def _crop_and_resize(image, zoom, image_size):
        """Crop and resize an image.

        Uses center cropping.

        Args:
            image (tensor): an input image tensor
            zoom (float): a zoom factor
            image_size ((int, int)): a desired output image size

        Returns:
            tensor: an outpu timage tensor
        """
        x1 = y1 = 0.5 - 0.5 * zoom  # scale centrally
        x2 = y2 = 0.5 + 0.5 * zoom
        boxes = tf.stack([y1, x1, y2, x2], axis=1)
        box_ind = [0]
        return tf.cast(tf.squeeze(
            tf.image.crop_and_resize(
                tf.expand_dims(image, 0),
                boxes,
                box_ind,
                image_size,
                method='nearest'
            )
        ), tf.uint8)

    @staticmethod
    def _blur(image, sigma):
        kernel = _gaussian_kernel_3d(sigma)
        # all preprocessing should run on the CPU
        with tf.device('/cpu:0'):
            blurred_image = tf.nn.depthwise_conv2d(
                tf.cast(tf.expand_dims(image, 0), tf.float32),
                kernel,
                [1, 1, 1, 1],
                padding='SAME',
                data_format="NHWC"
            )
        return blurred_image[0]

    @classmethod
    def _augment_example(cls, example):
        """Augment an example from the dataset.

        All augmentation functions are also be applied to the segmentation
        mask.

        Args:
            example (dict): a single example from the dataset.

        Returns:
            dict: an augmented example
        """
        image = example['image']
        mask = example['mask']

        image_size = image.shape.as_list()[0:2]

        # Add padding so we don't get black borders
        paddings = numpy.array(
            [[image_size[0] / 2, image_size[0] / 2],
             [image_size[1] / 2, image_size[1] / 2],
             [0, 0]], dtype=numpy.uint32)
        aug_image = tf.pad(image, paddings, mode='REFLECT')
        aug_mask = tf.pad(mask, paddings, mode='REFLECT')
        padded_image_size = [dim * 2 for dim in image_size]

        # Rotate
        angle = tf.random_uniform([1], -numpy.pi / 6, numpy.pi / 6)
        aug_image = tf.contrib.image.rotate(aug_image, angle)
        aug_mask = tf.contrib.image.rotate(aug_mask, angle)

        # Zoom
        zoom = tf.random_uniform([1], 0.85, 1.75)
        aug_image = cls._crop_and_resize(aug_image, zoom, padded_image_size)
        aug_mask = cls._crop_and_resize(aug_mask, zoom, padded_image_size)

        # Crop things back to original size
        aug_image = tf.image.central_crop(aug_image, central_fraction=0.5)
        aug_mask = tf.image.central_crop(aug_mask, central_fraction=0.5)

        # blur
        # Not used at the moment because it makes training hard
        # sigma = tf.random_uniform([1], 0.0, 1.0)
        # aug_image = cls._blur(aug_image, sigma)

        # Flip left right
        do_flip = tf.greater(tf.random_uniform([1], 0.0, 1.0)[0], 0.5)
        aug_image = tf.cond(
            do_flip,
            true_fn=lambda: tf.image.flip_left_right(aug_image),
            false_fn=lambda: aug_image,
        )
        aug_mask = tf.cond(
            do_flip,
            true_fn=lambda: tf.image.flip_left_right(aug_mask),
            false_fn=lambda: aug_mask,
        )

        # Flip up down
        do_flip = tf.greater(tf.random_uniform([1], 0.0, 1.0)[0], 0.5)
        aug_image = tf.cond(
            do_flip,
            true_fn=lambda: tf.image.flip_up_down(aug_image),
            false_fn=lambda: aug_image,
        )
        aug_mask = tf.cond(
            do_flip,
            true_fn=lambda: tf.image.flip_up_down(aug_mask),
            false_fn=lambda: aug_mask,
        )

        return {'image': aug_image, 'mask': aug_mask}

    @staticmethod
    def _decode_example(example_proto):
        """Decode an example from a TFRecord.

        Args:
            example_proto (tfrecord): a serialized tf record

        Returns:
            dict: an example from the dataset containing image and mask.
        """
        features = {
            "image/encoded": tf.FixedLenFeature(
                (), tf.string, default_value=""
            ),
            "image/segmentation/class/encoded": tf.FixedLenFeature(
                (), tf.string, default_value=""
            )
        }
        parsed_features = tf.parse_single_example(example_proto, features)
        image = tf.image.decode_jpeg(
            parsed_features["image/encoded"], channels=3)
        mask = tf.image.decode_png(
            parsed_features["image/segmentation/class/encoded"], channels=3)
        return {'image': image, 'mask': mask}

    @classmethod
    def _generate_multiscale_masks(cls, example, n_classes):
        """Generate masks at mulitple scales for training.

        The loss function compares masks at 4, 8, and 16x increases in scale.

        Args:
            example (dict): a single example from the dataset
            n_classes (int): the number of classes in the mask

        Returns
            (dict): the same example, but with additional mask data for each
                new resolution.
        """
        original_mask = example['mask']
        # Add the image to the placeholder
        image_size = example['image'].shape.as_list()[0:2]

        for scale in [4, 8, 16]:
            example['mask_%d' % scale] = tf.one_hot(
                cls._resize_fn(
                    original_mask,
                    list(map(lambda x: x // scale, image_size))
                )[:, :, 0],  # only need one channel
                depth=n_classes,
                dtype=tf.float32
            )
        return example

    @classmethod
    def scale_mask(cls, mask, scale, image_size, n_classes):
        return tf.one_hot(
            cls._resize_fn(
                mask,
                image_size,
            )[:, :, :, 0],  # only need one channel
            depth=n_classes,
            dtype=tf.float32
        )

    @classmethod
    def build(
            cls,
            filename,
            batch_size,
            image_size,
            n_classes,
            augment_images=True,
            repeat=True,
            prefetch=False,
            parallel_calls=1):
        """Build a TFRecord dataset.

        Args:
            filename (str): a .tfrecord file to read
            batch_size (int): batch size
            image_size (int): the desired image size of examples
            n_classes (int): the number of classes
            whitelist_threshold (float): the minimum fraction of whitelisted
                classes an example must contain to be used for training.

        Returns:
            dataset: a TFRecordDataset
        """
        logger.info('Creating dataset from: %s' % filename)
        dataset = tf.data.TFRecordDataset(filename)
        dataset = dataset.map(cls._decode_example,
                              num_parallel_calls=parallel_calls)
        dataset = dataset.map(lambda x: cls._resize_example(x, image_size),
                              num_parallel_calls=parallel_calls)
        if augment_images:
            dataset = dataset.map(cls._augment_example,
                                  num_parallel_calls=parallel_calls)
        dataset = dataset.map(cls._preprocess_example,
                              num_parallel_calls=parallel_calls)
        dataset = dataset.map(
            lambda x: cls._generate_multiscale_masks(x, n_classes),
            num_parallel_calls=parallel_calls
        )
        if repeat:
            dataset = dataset.repeat()

        dataset = dataset.batch(batch_size)
        if prefetch:
            dataset = dataset.prefetch(buffer_size=batch_size)
        return dataset
