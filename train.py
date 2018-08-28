"""Train an ICNet Model on ADE20K Data."""

import argparse
import keras
import logging
import sys

from image_segmentation.icnet import ICNetModelFactory
from image_segmentation.data_generator import ADE20KDatasetBuilder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('train')


def train(argv):
    """Train an ICNet model."""
    parser = argparse.ArgumentParser(
        description='Train an ICNet model.'
    )
    # Data options
    parser.add_argument(
        '-d', '--tfrecord-data', type=str, required=True,
        help='A TFRecord file containing images and segmentation masks.'
    )
    parser.add_argument(
        '-l', '--label-filename', type=str, required=True,
        help='A file containing a single label per line.'
    )
    parser.add_argument(
        '-s', '--image-size', type=int, default=768,
        help=('The pixel dimension of model input and output. Images '
              'will be square.')
    )
    parser.add_argument(
        '-a', '--augment-images', type=bool, default=True,
        help='turn on image augmentation.'
    )
    parser.add_argument(
        '-w', '--whitelist-labels', type=str,
        help=('A pipe | separated list of object labels to whitelist. To see a'
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
    # Training options
    parser.add_argument(
        '-b', '--batch-size', type=int, default=8,
        help='The training batch_size.'
    )
    parser.add_argument(
        '--lr', type=float, default=0.001, help='The learning rate.'
    )
    parser.add_argument(
        '-n', '--num-steps', type=int, default=1000,
        help='Number of training steps to perform'
    )
    parser.add_argument(
        '-o', '--output', type=str, required=True,
        help='An output file to save the trained model.')
    parser.add_argument(
        '--checkpoint', type=str,
        help='A Keras model checkpoint to load and continue training.'
    )

    args = parser.parse_args(argv)

    if args.list_labels:
        logger.info('Labels:')
        labels = ''
        for label in ADE20KDatasetBuilder.load_class_labels(
                args.label_filename):
            labels += '%s\n' % label
        logger.info(labels)
        sys.exit()

    whitelist_labels = None
    if args.whitelist_labels:
        whitelist_labels = args.whitelist_labels.split('|')

    dataset_builder = ADE20KDatasetBuilder(
        args.label_filename,
        whitelist_labels=whitelist_labels
    )

    dataset = dataset_builder.build(
        args.tfrecord_data,
        batch_size=args.batch_size,
        whitelist_threshold=args.whitelist_threshold,
        image_size=(args.image_size, args.image_size),
        augment_images=args.augment_images
    )

    iterator = dataset.make_one_shot_iterator()
    example = iterator.get_next()

    icnet = ICNetModelFactory.build(
        args.image_size,
        dataset_builder.n_classes,
        weights_path=args.checkpoint,
        train=True,
        input_tensor=example['image']
    )

    optimizer = keras.optimizers.Adam(lr=args.lr)
    icnet.compile(
        optimizer,
        loss=keras.losses.categorical_crossentropy,
        loss_weights=[1.0, 0.4, 0.16],
        metrics=['categorical_accuracy'],
        target_tensors=[
            example['mask_4'], example['mask_8'], example['mask_16']
        ]
    )

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            args.output,
            verbose=0,
            mode='auto',
            period=1
        ),
    ]

    icnet.fit(
        steps_per_epoch=args.num_steps,
        epochs=1,
        callbacks=callbacks,
    )

if __name__ == '__main__':
    train(sys.argv[1:])
