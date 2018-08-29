"""Summary.

Attributes:
    logger (TYPE): Description
"""
import numpy
import logging

import tensorflow as tf
from tensorflow.python.lib.io import file_io

logger = logging.getLogger('data_generator')


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
        boxes = numpy.array([[y1, x1, y2, x2]], dtype=numpy.float32)
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
        angle = numpy.random.uniform(-numpy.pi / 6, numpy.pi / 6)
        aug_image = tf.contrib.image.rotate(aug_image, angle)
        aug_mask = tf.contrib.image.rotate(aug_mask, angle)

        # Zoom
        zoom = numpy.random.uniform(0.75, 1.75)
        aug_image = cls._crop_and_resize(aug_image, zoom, padded_image_size)
        aug_mask = cls._crop_and_resize(aug_mask, zoom, padded_image_size)

        # Crop things back to original size
        aug_image = tf.image.central_crop(aug_image, central_fraction=0.5)
        aug_mask = tf.image.central_crop(aug_mask, central_fraction=0.5)
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
    def build(
            cls,
            filename,
            batch_size,
            image_size,
            n_classes,
            augment_images=True):
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
        dataset = dataset.map(cls._decode_example)
        dataset = dataset.map(lambda x: cls._resize_example(x, image_size))
        if augment_images:
            dataset = dataset.map(cls._augment_example)
        dataset = dataset.map(cls._preprocess_example)
        dataset = dataset.map(
            lambda x: cls._generate_multiscale_masks(x, n_classes)
        )
        dataset = dataset.repeat()
        dataset = dataset.batch(batch_size)
        return dataset
