import os
import glob
import numpy
import PIL.Image
import random
import gc
import skimage.transform
import skimage.filters
import logging
import keras.utils

logger = logging.getLogger()


class ADE20KGenerator(keras.utils.Sequence):
    """A data generator that implements the Keras Sequence protocol.

    This generator feeds images from the ADE20K data set maintained by MIT.

    More information on the dataset can be found here:
        http://groups.csail.mit.edu/vision/datasets/ADE20K/

    """

    def __init__(
            self,
            root_directory,
            mode='training',
            batch_size=1,
            image_size=(384, 384),
            whitelist_labels=None,
            whitelist_threshold=0.7,
            augment_images=True):
        """Initalize the generator.

        Args:
            root_directory (str): the root directory containing dataset images
            mode (str): data mode can be 'training' or 'validation'
            batch_size (int, optional): output 'batch_size' images at at time
            image_size ((int, int)): the output size of each image
            whitelist_labels (List[str]): a list of object labels each image
                must contain to be in the final dataset
            whitelist_threshold (float, optional): for an image to be included,
                the fraction of whitelisted labels it contains must be greater
                than 'whitelist_threshold'
            augment_images (bool): if true, images are augmented randomly.
        """
        self.mode = mode
        self.batch_size = batch_size
        self.image_size = image_size
        self.root_directory = root_directory
        self.augment_images = augment_images

        # Load the class labels from the index file
        self.class_labels = self.load_class_labels(self.root_directory)
        self.n_classes = len(self.class_labels)
        self.whitelist_labels = whitelist_labels
        self.whitelist_threshold = whitelist_threshold
        # If a whitelist is specified, get the label index for each.
        if whitelist_labels:
            self.whitelist_labels.insert(0, 'none')
            self.whitelist_labels_index = self._find_whitelist_label_indices(
                self.whitelist_labels
            )
            self.n_classes = len(self.whitelist_labels)
            logger.info('Whitelist has %d classes total.' % self.n_classes)

        # It'd be nice to use iglob and just get an iterator
        # but we need this to be sorted so the pairs match up
        self.image_path_list = sorted(glob.glob(
            os.path.join(root_directory, 'images/', mode, '*.jpg'),
            recursive=True
        ))
        self.mask_path_list = sorted(glob.glob(
            os.path.join(root_directory, 'annotations/', mode, '*.png'),
            recursive=True
        ))

        # If we only want certain labels, loop through all of the
        # annotations and grab just images we want.
        keep_image_path = []
        keep_mask_path = []
        if self.whitelist_labels:
            logging.info('Scanning for images containing whitelisted labels.')
            for k, mask_path in enumerate(self.mask_path_list):
                mask = self._load_mask(mask_path)
                unique_labels = numpy.unique(mask)
                num_found = numpy.intersect1d(
                    unique_labels,
                    self.whitelist_labels_index
                ).size
                if (float(num_found) / len(self.whitelist_labels_index) >=
                        self.whitelist_threshold):
                    keep_image_path.append(self.image_path_list[k])
                    keep_mask_path.append(mask_path)
            self.image_path_list = keep_image_path
            self.mask_path_list = keep_mask_path
            logger.info('Keeping %d images based on labels.' %
                        len(self.image_path_list))

        # Setup some placeholders
        self._init_placeholders()

    def _init_placeholders(self):
        """Initialize placeholders.

        During training, loss is computed by comparing pixel masks at multiple
        scales. We need tensors to represent outputs for each.
        """
        self.X = numpy.zeros(
            (self.batch_size, self.image_size[0], self.image_size[1], 3)
        )
        self.Y1 = numpy.zeros((
            self.batch_size,
            self.image_size[0] // 4,
            self.image_size[1] // 4,
            self.n_classes
        ))
        self.Y2 = numpy.zeros((
            self.batch_size,
            self.image_size[0] // 8,
            self.image_size[1] // 8,
            self.n_classes
        ))
        self.Y3 = numpy.zeros((
            self.batch_size,
            self.image_size[0] // 16,
            self.image_size[1] // 16,
            self.n_classes
        ))

    def set_image_size(self, image_size):
        """Set the image size of the generator.

        Args:
            image_size (int): the height and width dimesion of the image
        """
        self.image_size = image_size
        self._init_placeholders()

    def set_batch_size(self, batch_size):
        """Set the batch size.

        Args:
            batch_size (int): Description
        """
        self.batch_size = batch_size
        self._init_placeholders()

    @staticmethod
    def load_class_labels(root_directory):
        """Load class labels.

        Assumes the data directory is left unchanged from the original zip.

        Args:
            root_directory (str): the dataset's root directory

        Returns:
            arr: an array of class labels
        """
        class_labels = []
        header = True
        with open(os.path.join(root_directory, 'objectInfo150.txt')) as file:
            for line in file.readlines():
                if header:
                    class_labels.append('none')
                    header = False
                    continue
                line = line.rstrip()
                label = line.split('\t')[-1]
                class_labels.append(label)
        return numpy.array(class_labels)

    def _find_whitelist_label_indices(self, whitelist_labels):
        """Map whitelist labels to indices.

        Args:
            whitelist_labels (List[str]): a list of whitelisted labels

        Returns:
            arr: an array of label indices
        """
        index = []
        for label in whitelist_labels:
            for idx, class_label in enumerate(self.class_labels):
                if label == class_label:
                    index.append(idx)
        return numpy.array(index)

    def _load_mask(self, mask_path):
        """Load an image segmentation mask.

        Args:
            mask_path (str): absolute path to the mask image.

        Returns:
            arr: an array of mask data
        """
        return numpy.array(
            PIL.Image.open(mask_path).resize(self.image_size)
        ).astype('float')

    def _load_image(self, image_path):
        """Load an image.

        Also applies preprocessing, normalizing pixel values so they fall
        between -1 and 1.

        Args:
            image_path (str): absolute path to an image.

        Returns:
            arr: an array containing image data in [H, W, C] format.
        """
        image = numpy.array(
            PIL.Image.open(image_path).resize(self.image_size)
        )
        # Preprocessing
        image = (image - 127.5) / 255.0
        return image

    def __len__(self):
        """Return the total number of batches in the dataset..

        Returns:
            int: total number of batches in the dataset.
        """
        return len(self.image_path_list) // self.batch_size

    @staticmethod
    def _augment_image(image, mask):
        """Augment an image.

        Applies random rotation, zooming, and blurring. Images as masks needs
        to be augmented in the same way.

        Args:
            image (arr): an image to augment.
            mask (arr): a corresponding segmentation mask

        Return:
            augmented_image (arr): an augmented image
            augmented_masK (arr): an augmented segmentation mask
        """
        # Rotate
        angle = numpy.random.uniform(-30, 30)
        aug_image = skimage.transform.rotate(image, angle, mode='reflect')
        aug_mask = skimage.transform.rotate(
            mask, angle, mode='reflect', order=0
        )

        # Zoom
        zoom = numpy.random.uniform(0.75, 1.75)
        transform = skimage.transform.AffineTransform(scale=(zoom, zoom))
        aug_image = skimage.transform.warp(
            aug_image, transform, mode='reflect'
        )
        aug_mask = skimage.transform.warp(
            aug_mask, transform, mode='reflect', order=0
        )

        # Blur
        sigma = numpy.random.uniform(0, 1)
        aug_image = skimage.filters.gaussian(aug_image, sigma)
        return aug_image, aug_mask

    def __getitem__(self, i):
        """Get a batch.

        Args:
            i (int): the indext of the batch.

        Returns:
            arr: a batch of images [B, H, W, C]
        """
        batch_start = i * self.batch_size
        batch_stop = (i + 1) * self.batch_size
        image_paths = self.image_path_list[batch_start: batch_stop]
        mask_paths = self.mask_path_list[batch_start: batch_stop]
        paths = zip(image_paths, mask_paths)
        for n, (image_path, mask_path) in enumerate(paths):
            image = self._load_image(image_path)
            if image.shape[-1] != 3:
                continue

            mask = self._load_mask(mask_path)
            if self.whitelist_labels:
                # The orignal masks will include all labels, even ones that
                # aren't whitelisted. We need to create a new mask with just
                # the labels we care about, re-indexed from 0 to n_classes.
                new_mask = numpy.zeros(mask.shape)
                for new_label, old_label in enumerate(self.whitelist_labels_index):  # NOQA
                    idx = numpy.where(mask == old_label)
                    new_mask[idx] = new_label
                mask = new_mask.copy()

            # Do some augmentation
            if self.augment_images:
                # Rotate
                image, mask = self._augment_image(image, mask)

            # Add the image to the placeholder
            self.X[n] = image
            self.Y1[n] = keras.utils.to_categorical(
                skimage.transform.resize(
                    mask,
                    list(map(lambda x: x // 4, mask.shape)),
                    order=0),
                self.n_classes
            )
            self.Y2[n] = keras.utils.to_categorical(
                skimage.transform.resize(
                    mask,
                    list(map(lambda x: x // 8, mask.shape)),
                    order=0),
                self.n_classes
            )
            self.Y3[n] = keras.utils.to_categorical(
                skimage.transform.resize(
                    mask,
                    list(map(lambda x: x // 16, mask.shape)),
                    order=0),
                self.n_classes
            )
        return self.X, [self.Y1, self.Y2, self.Y3]

    def on_epoch_end(self):
        """Shuffle the dataset for the next epoch."""
        file_pairs = list(zip(self.image_path_list, self.mask_path_list))
        random.shuffle(file_pairs)
        self.image_path_list, self.mask_path_list = zip(*file_pairs)

        # Fix memory leak (Keras bug)
        gc.collect()
