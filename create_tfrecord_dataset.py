# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import argparse
from functools import partial
import logging
import os
import io
import numpy
import sys

import PIL.Image
import tensorflow as tf
from tensorflow.python.lib.io import file_io

from image_segmentation import build_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('create_tfrecord_dataset')


def main(argv):
    parser = argparse.ArgumentParser(
        description='Convert the ADE20K Challenge dataset to tfrecords'
    )

    parser.add_argument(
        '-i', '--image-dir', type=str, required=True,
        help='Folder containing trainng images'
    )
    parser.add_argument(
        '-a', '--annotation-dir', type=str, required=True,
        help='Folder containing annotations for training images'
    )
    parser.add_argument(
        '-o', '--output', type=str, required=True,
        help='Path to save converted tfrecord of Tensorflow example'
    )
    parser.add_argument(
        '-l', '--label-filename', type=str, required=True,
        help='A file containing a single label per line.'
    )
    parser.add_argument(
        '-w', '--whitelist-labels', type=str,
        help=('A pipe | separated list of object labels to whitelist. '
              'categories can be merged by seperating them by : '
              'e.g. "person|car:truck:van|pavement". To see a'
              ' full list of allowed labels run with  --list-labels.')
    )
    parser.add_argument(
        '-t', '--whitelist-threshold', type=float, default=0.7,
        help=('The fraction of whitelisted labels an image must contain to be '
              'used for training.')
    )
    parser.add_argument(
        '--list-labels', action='store_true',
        help='If true, print a full list of object labels.'
    )

    args = parser.parse_args(argv)

    # Load the class labels
    class_labels = _load_class_labels(args.label_filename)
    n_classes = len(class_labels)
    if args.list_labels:
        logger.info('Labels:')
        labels = ''
        for label in class_labels:
            labels += '%d, %s\n' % label
        logger.info(labels)
        sys.exit()

    # If a whitelist is provided, get a list of mask indices that correspond
    # to allowed labels
    whitelist_labels = None
    whitelist_indices = None
    if args.whitelist_labels:
        whitelist_labels = _parse_whitelist_labels(args.whitelist_labels)

        # add a 'none' class with a label of 0
        whitelist_labels.insert(0, ['none'])
        whitelist_indices = _find_whitelist_indices(
            class_labels, whitelist_labels)

        whitelist_filename = os.path.join(
            os.path.dirname(args.output), 'labels.txt')
        _save_whitelist_labels(whitelist_filename, whitelist_labels)
        n_classes = len(whitelist_labels)

    _create_tfrecord_dataset(
        args.image_dir,
        args.annotation_dir,
        args.output,
        n_classes,
        whitelist_indices=whitelist_indices,
        whitelist_threshold=args.whitelist_threshold
    )


def _parse_whitelist_labels(whitelist):
    parsed = whitelist.split('|')
    parsed = [category.split(':') for category in parsed]
    return parsed


def _save_whitelist_labels(whitelist_filename, labels):
    with open(whitelist_filename, 'w') as wfid:
        header = 'idx\tlabel\n'
        wfid.write(header)
        for idx, label_set in enumerate(labels):
            label = label_set[0].split(',')[0]
            wfid.write('%d\t%s\n' % (idx, label))
    print("Saved")


def _load_class_labels(label_filename):
    """Load class labels.

    Assumes the data directory is left unchanged from the original zip.

    Args:
        root_directory (str): the dataset's root directory

    Returns:
        List[(int, str)]: a list of class ids and labels
    """
    class_labels = []
    header = True
    with file_io.FileIO(label_filename, mode='r') as file:
        for line in file.readlines():
            if header:
                class_labels.append((0, 'none'))
                header = False
                continue
            line = line.rstrip()
            line = line.split('\t')
            label = line[-1]
            label_id = int(line[0])
            class_labels.append((label_id, label))
    return class_labels


def _find_whitelist_indices(class_labels, whitelist_labels):
    """Map whitelist labels to indices.

    Args:
        whitelist (List[str]): a list of whitelisted labels

    Returns:
        List[Set]: a list of sets containing index labels
    """
    index = []
    for label_set in whitelist_labels:
        index_set = []
        for label in label_set:
            for class_id, class_label in class_labels:
                if label == class_label:
                    index_set.append(class_id)
        index.append(index_set)
    return index


def _filter_whitelabel_classes(
        filenames,
        whitelist,
        whitelist_threshold,
        whitelist_size=None):
    w_size = whitelist_size or len(whitelist)
    mask = numpy.array(PIL.Image.open(filenames[-1]))
    unique_classes = numpy.unique(mask)
    num_found = numpy.intersect1d(unique_classes, whitelist).size
    if float(num_found) / w_size >= whitelist_threshold:
        return True
    return False


def _relabel_mask(seg_data, whitelist_indices):
    # Read the data into a numpy array.
    mask = numpy.array(PIL.Image.open(io.BytesIO(seg_data)))
    # Relabel each pixel
    new_mask = numpy.zeros(mask.shape)
    for new_label, old_label_set in enumerate(whitelist_indices):
        idx = numpy.where(numpy.isin(mask, old_label_set))
        new_mask[idx] = new_label
    # Convert the new mask back to an image.
    seg_img = PIL.Image.fromarray(new_mask.astype('uint8')).convert('RGB')
    # Save the new image to a PNG byte string.
    byte_buffer = io.BytesIO()
    seg_img.save(byte_buffer, format='png')
    byte_buffer.seek(0)
    return byte_buffer.read()


def _create_tfrecord_dataset(
        image_dir,
        segmentation_mask_dir,
        output_filename,
        n_classes,
        whitelist_indices=None,
        whitelist_threshold=0.5):
    """Convert the ADE20k dataset into into tfrecord format.

    Args:
        dataset_split: Dataset split (e.g., train, val).
        dataset_dir: Dir in which the dataset locates.
        dataset_label_dir: Dir in which the annotations locates.
    Raises:
        RuntimeError: If loaded image and label have different shape.
    """
    # Get all of the image and segmentation mask file names
    img_names = tf.gfile.Glob(os.path.join(image_dir, '*.jpg'))
    seg_names = []
    for f in img_names:
        # get the filename without the extension
        basename = os.path.basename(f).split('.')[0]
        # cover its corresponding *_seg.png
        seg = os.path.join(segmentation_mask_dir, basename + '.png')
        seg_names.append(seg)

    # If a whitelist has been provided, loop over all of the segmentation
    # masks and find only the images that contain enough classes.
    kept_files = zip(img_names, seg_names)
    if whitelist_indices is not None:
        # Flatten the whitelist because some categories have been merged
        # but make sure to use the orginal list size when
        # computing the threshold.
        flat_whitelist = numpy.array(
            [idx for idx_set in whitelist_indices for idx in idx_set]
        ).astype('uint8')
        merged_whitelist_size = len(whitelist_indices)
        filter_fn = partial(
            _filter_whitelabel_classes,
            whitelist=flat_whitelist,
            whitelist_threshold=whitelist_threshold,
            whitelist_size=merged_whitelist_size
        )
        kept_files = list(filter(filter_fn, kept_files))
        logger.info(
            'Found %d images after whitelist filtereing.' % len(kept_files))
    num_images = len(kept_files)
    image_reader = build_data.ImageReader('jpeg', channels=3)
    label_reader = build_data.ImageReader('png', channels=1)

    with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
        for idx, (image_filename, seg_filename) in enumerate(kept_files):
            if idx % 100 == 0:
                logger.info('Converting image %d of %d.' % (idx, num_images))
            # Read the image.
            image_data = tf.gfile.FastGFile(image_filename, 'rb').read()
            height, width = image_reader.read_image_dims(image_data)
            # Read the semantic segmentation annotation.
            seg_data = tf.gfile.FastGFile(seg_filename, 'rb').read()
            # If there is a whitelist, we need to relabel all of the
            # mask classes so that only the whitelisted labels are present.
            if whitelist_indices is not None:
                seg_data = _relabel_mask(seg_data, whitelist_indices)
            seg_height, seg_width = label_reader.read_image_dims(seg_data)
            if height != seg_height or width != seg_width:
                raise RuntimeError(
                    'Shape mismatched between image and label.')
            # Convert to tf example.
            example = build_data.image_seg_to_tfexample(
                image_data, image_filename, height, width, seg_data)
            tfrecord_writer.write(example.SerializeToString())


if __name__ == '__main__':
    main(sys.argv[1:])
