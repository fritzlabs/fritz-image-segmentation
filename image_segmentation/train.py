"""Train an ICNet Model on ADE20K Data."""

import argparse
import keras
import logging
import time
import sys
import struct
import os
from tensorflow.python.lib.io import file_io
import tensorflow as tf
from image_segmentation.icnet import ICNetModelFactory
from image_segmentation.data_generator import ADE20KDatasetBuilder
from image_segmentation import dali_config
from google.cloud import storage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('train')


def _summarize_arguments(args):
    """Summarize input arguments to ICNet model training.

    Args:
        args:
    """

    logger.info('ICNet Model training Parameters')
    logger.info('-------------------------------')
    for key, value in vars(args).items():
        logger.info('    {key}={value}'.format(key=key, value=value))


def _build_parser(argv):
    parser = argparse.ArgumentParser(
        description='Train an ICNet model.'
    )
    # Data options
    parser.add_argument(
        '-d', '--data', nargs='+', required=True,
        help='A TFRecord file containing images and segmentation masks.'
    )
    parser.add_argument(
        '--tfindex-files', nargs='+',
        help='TFIndex file for dali pipeline. If not included, will be built'
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
        '-a', '--alpha', type=float, default=1.0,
        help='The width multiplier for the network'
    )
    parser.add_argument(
        '--augment-images', type=bool, default=True,
        help='turn on image augmentation.'
    )
    parser.add_argument(
        '--add-noise', action='store_true',
        help='Add gaussian noise to training.'
    )
    parser.add_argument(
        '--use-dali', action='store_true',
        help='turn on image augmentation.'
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
        '--steps-per-epoch', type=int, default=100,
        help='Number of training steps to perform between model checkpoints'
    )
    parser.add_argument(
        '-o', '--output',
        help='An output file to save the trained model.')
    parser.add_argument(
        '--gpu-cores', type=int, default=1,
        help='Number of GPU cores to run on.')
    parser.add_argument(
        '--fine-tune-checkpoint', type=str,
        help='A Keras model checkpoint to load and continue training.'
    )
    parser.add_argument(
        '--gcs-bucket', type=str,
        help='A GCS Bucket to save models too.'
    )
    parser.add_argument(
        '--parallel-calls', type=int, default=1,
        help='Number of parallel calss to preprocessing to perform.'
    )
    parser.add_argument(
        '--model-name', type=str, required=True,
        help='Short name separated by underscores'
    )

    return parser.parse_known_args()


def _prepare_dataset(args, n_classes):
    dataset = ADE20KDatasetBuilder.build(
        args.data,
        n_classes=n_classes,
        batch_size=args.batch_size,
        image_size=(args.image_size, args.image_size),
        augment_images=False,
        parallel_calls=args.parallel_calls,
        prefetch=True,
    )

    iterator = dataset.make_one_shot_iterator()
    example = iterator.get_next()

    return {
        'input': example['image'],
        'mask_4': example['mask_4'],
        'mask_8': example['mask_8'],
        'mask_16': example['mask_16'],
    }


def build_tfindex_file(tfrecord_file, tfindex_file):
    """Builds a tfindex file used by DALI from a tfrecord file.

    Args:
        tfrecord_file: Path to TFRecord file.
        tfindex_file: output file to write to.
    """
    tfrecord_fp = open(tfrecord_file, 'rb')
    idx_fp = open(tfindex_file, 'w')

    while True:
        current = tfrecord_fp.tell()
        try:
            # length
            byte_len = tfrecord_fp.read(8)
            if byte_len == '':
                break
            # crc
            tfrecord_fp.read(4)
            proto_len = struct.unpack('q', byte_len)[0]
            # proto
            tfrecord_fp.read(proto_len)
            # crc
            tfrecord_fp.read(4)
            idx_fp.write(str(current) + ' ' +
                         str(tfrecord_fp.tell() - current) + '\n')
        except Exception:
            print("Not a valid TFRecord file")
            break

    tfrecord_fp.close()
    idx_fp.close()


def _prepare_dali(args, n_classes):
    if args.gpu_cores > 1:
        logger.error(
            'Have not built in support for more than one GPU at the moment.'
        )
        sys.exit(1)

    # non NVIDIA cloud environments will not have dali, so we
    # have to do the import here.
    from image_segmentation.dali_pipeline import CommonPipeline
    import nvidia.dali.plugin.tf as dali_tf

    batch_size = args.batch_size
    image_size = args.image_size
    device_id = 0
    storage_client = storage.Client()
    filenames = []

    for filename in args.data:
        if filename.startswith('gs://'):
            parts = filename[5:].split('/')
            bucket_name, blob_name = parts[0], '/'.join(parts[1:])
            bucket = storage_client.get_bucket(bucket_name)
            blob = bucket.blob(blob_name)
            download_filename = os.path.basename(blob_name)
            blob.download_to_filename(download_filename)
            filenames.append(download_filename)
        else:
            filenames.append(filename)

    tfindex_files = args.tfindex_files or []
    if not tfindex_files:
        for path in filenames:
            tfindex_file = path.split('.')[0] + '.tfindex'
            build_tfindex_file(path, tfindex_file)
            logger.info('Created tfindex file: {input} -> {output}'.format(
                input=path,
                output=tfindex_file
            ))
            tfindex_files.append(tfindex_file)

    config = dali_config.DaliConfig()
    config.summarize()

    pipe = CommonPipeline(
        args.batch_size,
        args.parallel_calls,
        device_id,
        args.image_size,
        filenames,
        tfindex_files,
        config
    )
    pipe.build()

    daliop = dali_tf.DALIIterator()
    with tf.device('/gpu:0'):
        results = daliop(
            serialized_pipeline=pipe.serialize(),
            shape=[args.batch_size, args.image_size, args.image_size, 3],
            label_type=tf.int64,
        )

    input_tensor = results.batch

    results.label.set_shape([batch_size, image_size, image_size, 3])
    mask = results.label
    new_shape = [image_size / 4, image_size / 4]
    mask_4 = ADE20KDatasetBuilder.scale_mask(mask, 4, new_shape, n_classes)
    new_shape = [image_size / 8, image_size / 8]
    mask_8 = ADE20KDatasetBuilder.scale_mask(mask, 8, new_shape, n_classes)
    new_shape = [image_size / 16, image_size / 16]
    mask_16 = ADE20KDatasetBuilder.scale_mask(mask, 16, new_shape, n_classes)

    return {
        'input': input_tensor,
        'mask_4': mask_4,
        'mask_8': mask_8,
        'mask_16': mask_16,
    }


def train(argv):
    """Train an ICNet model."""

    args, unknown = _build_parser(argv)
    _summarize_arguments(args)

    class_labels = ADE20KDatasetBuilder.load_class_labels(
        args.label_filename)
    if args.list_labels:
        logger.info('Labels:')
        labels = ''
        for label in class_labels:
            labels += '%s\n' % label
        logger.info(labels)
        sys.exit()

    n_classes = len(class_labels)

    if args.use_dali:
        data = _prepare_dali(args, n_classes)
    else:
        data = _prepare_dataset(args, n_classes)

    if args.add_noise:
        logger.info('Adding gaussian noise to input tensor.')
        noise = tf.random_normal(shape=tf.shape(data['input']),
                                 mean=0.0,
                                 stddev=0.07,
                                 dtype=tf.float32)
        data['input'] = data['input'] + noise

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    config = tf.ConfigProto(gpu_options=gpu_options)
    sess = tf.Session(config=config)
    keras.backend.set_session(sess)

    if args.gpu_cores > 1:
        with tf.device('/CPU:0'):
            icnet = ICNetModelFactory.build(
                args.image_size,
                n_classes,
                weights_path=args.fine_tune_checkpoint,
                train=True,
                input_tensor=data['input'],
                alpha=args.alpha,
            )

        gpu_icnet = keras.utils.multi_gpu_model(icnet, gpus=args.cores)
        gpu_icnet.__setattr__('callback_model', icnet)
        model = gpu_icnet
    else:
        with tf.device('/GPU:0'):
            model = ICNetModelFactory.build(
                args.image_size,
                n_classes,
                weights_path=args.fine_tune_checkpoint,
                train=True,
                input_tensor=data['input'],
                alpha=args.alpha,
            )

    optimizer = keras.optimizers.Adam(lr=args.lr)
    model.compile(
        optimizer,
        loss=keras.losses.categorical_crossentropy,
        loss_weights=[1.0, 0.4, 0.16],
        metrics=['categorical_accuracy'],
        target_tensors=[
            data['mask_4'], data['mask_8'], data['mask_16']
        ]
    )

    if not args.output:
        output_filename_fmt = '{model_name}_{size}x{size}_{alpha}_{time}.h5'
        filename = output_filename_fmt.format(
            model_name=args.model_name,
            size=args.image_size,
            alpha=str(args.alpha).replace('0', '').replace('.', ''),
            time=int(time.time())
        )
    else:
        filename = args.output

    print("=======================")
    print("Output file name: {name}".format(name=filename))
    print("=======================")

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filename,
            verbose=0,
            mode='auto',
            period=1
        ),
    ]

    if args.gcs_bucket:
        callbacks.append(SaveCheckpointToGCS(filename, args.gcs_bucket))

    model.fit(
        steps_per_epoch=args.steps_per_epoch,
        epochs=int(args.num_steps / args.steps_per_epoch) + 1,
        callbacks=callbacks,
    )


class SaveCheckpointToGCS(keras.callbacks.Callback):
    """A callback to save local model checkpoints to GCS."""

    def __init__(self, local_filename, gcs_filename):
        """Save a checkpoint to GCS.

        Args:
            local_filename (str): the path of the local checkpoint
            gcs_filename (str): the GCS bucket to save the model to
        """
        self.gcs_filename = gcs_filename
        self.local_filename = local_filename

    @staticmethod
    def _copy_file_to_gcs(job_dir, file_path):
        gcs_url = os.path.join(job_dir, file_path)
        logger.info('Saving models to GCS: %s' % gcs_url)
        with file_io.FileIO(file_path, mode='rb') as input_f:
            with file_io.FileIO(gcs_url, mode='w+') as output_f:
                output_f.write(input_f.read())

    def on_epoch_end(self, epoch, logs={}):
        """Save model to GCS on epoch end.

        Args:
            epoch (int): the epoch number
            logs (dict, optional): logs dict
        """
        basename = os.path.basename(self.local_filename)
        self._copy_file_to_gcs(self.gcs_filename, basename)


if __name__ == '__main__':
    train(sys.argv[1:])
