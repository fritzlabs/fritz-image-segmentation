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

    whitelist_indices = None
    whitelist_labels = None

    def __init__(self, label_filename, whitelist_labels=None):
        """Build a ADE20K TFRecordDataset.

        If a whitelist is provided, masks will be relabeled so that only pixels
        with those labels are shown. The index of each label will be reset
        and a 'none' label with index 0 will be used for any pixel that does
        not contain a label in the whitelist.

        Args:
            label_filename (str): a filename for class labels
            whitelist (List[str], optional): a list of allowed class labels
        """
        self.class_labels = self.load_class_labels(label_filename)
        self.n_classes = len(self.class_labels)
        if whitelist_labels:
            self.whitelist_labels = whitelist_labels
            # add a 'none' class with a label of 0
            self.whitelist_labels.insert(0, 'none')
            self.whitelist_indices = self._find_whitelist_indices(
                whitelist_labels)
            self.n_classes = len(self.whitelist_labels)

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
                    class_labels.append('none')
                    header = False
                    continue
                line = line.rstrip()
                label = line.split('\t')[-1]
                class_labels.append(label)
        return numpy.array(class_labels)

    def _find_whitelist_indices(self, whitelist_labels):
        """Map whitelist labels to indices.

        Args:
            whitelist (List[str]): a list of whitelisted labels

        Returns:
            arr: an array of label indices
        """
        index = []
        for label in whitelist_labels:
            for idx, class_label in enumerate(self.class_labels):
                if label == class_label:
                    index.append(idx)
        return numpy.array(index).astype('uint8')

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

    @staticmethod
    def _filter_whitelabeled_classes(example, whitelist, whitelist_threshold):
        """Filter examples based on whitelabeled classes.

        Only images containing enough classes from a whitelist will be
        passed on for training.

        In order for an image to be passed along for training, the fraction of
        labels from the whitelist it contains must be greater than
        `whitelist_threshold` .

        Args:
            example (dict): a single example from the dataset containing image
                and mask
            whitelist (List[int]): a list allowed of class label indexes
            whitelist_threshold (float): the minimum fraction of whitelisted
                classes an example must contain to be used for training.

        Returns:
            bool: true if the image can be used for training, false otherwise.
        """
        mask = example['mask']
        # Find unique classes in label mask
        unique_classes = tf.unique(tf.reshape(mask, [tf.size(mask)]))
        # Count the number of labels from the whitelist we are missing
        num_missing = tf.cast(
            tf.size(tf.setdiff1d(whitelist, unique_classes.y).out), tf.float32
        )
        total = len(whitelist)
        overlap = 1.0 - num_missing / total
        # If the mask contains more than whitelist_thresh fraction of the
        # whitelisted labels, include the example, otherwise skip.
        return tf.greater_equal(overlap, whitelist_threshold)

    @staticmethod
    def _relabel_mask(example, whitelist):
        """Relabel the mask so it includes only whitelisted labels.

        The orignal masks will include all labels, even ones that
        aren't whitelisted. We need to create a new mask with just
        the labels we care about, re-indexed from 0 to n_classes.

        Args:
            example (dict): a single example from the dataset
            whitelist (List[int]): a list of allowed label
                indexes

        Returns:
            dict: a single example with the segmentation mask re-labeled
        """
        mask = example['mask']
        # Find the indices of each whitelist label
        # and pick a new value to map them to
        new_mask = tf.reshape(mask, [tf.size(mask)])
        idx = tf.where(tf.equal(new_mask, [[el] for el in whitelist]))
        indices = tf.expand_dims(idx[:, 1], 1)
        updates = idx[:, 0]
        shape = new_mask.shape
        # Create a new tensor with the updated labels
        new_mask = tf.scatter_nd(
            indices,
            updates,
            shape
        )
        # Go back to the original shape
        example['mask'] = tf.reshape(new_mask, mask.shape)
        return example

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

    def build(
            self,
            filename,
            batch_size,
            image_size,
            whitelist_threshold=0.75,
            augment_images=True):
        """Build a TFRecord dataset.

        Args:
            filename (str): a .tfrecord file to read
            batch_size (int): batch size
            image_size (int): the desired image size of examples
            whitelist_threshold (float): the minimum fraction of whitelisted
                classes an example must contain to be used for training.

        Returns:
            dataset: a TFRecordDataset
        """
        logger.info('Creating dataset from: %s' % filename)
        dataset = tf.data.TFRecordDataset(filename)
        dataset = dataset.map(self._decode_example)
        dataset = dataset.map(lambda x: self._resize_example(x, image_size))
        if self.whitelist_indices is not None:
            dataset = dataset.filter(
                lambda x: self._filter_whitelabeled_classes(
                    x, self.whitelist_indices, whitelist_threshold
                )
            )
            dataset = dataset.map(
                lambda x: self._relabel_mask(x, self.whitelist_indices)
            )
        if augment_images:
            dataset = dataset.map(self._augment_example)
        dataset = dataset.map(self._preprocess_example)
        dataset = dataset.map(
            lambda x: self._generate_multiscale_masks(x, self.n_classes)
        )
        dataset = dataset.repeat()
        dataset = dataset.batch(batch_size)
        return dataset
