import argparse
import sys

import coremltools
import keras

from image_segmentation.icnet import ICNetModelFactory


def convert(argv):
    parser = argparse.ArgumentParser(
        description='Convert a Keras ICNet model to Core ML'
    )
    parser.add_argument(
        'keras_checkpoint', nargs='?', type=str,
        help='a Keras model checkpoint to load and convert.'
    )
    parser.add_argument(
        '--alpha', type=float, required=True,
        help='The width paramter of the network.')
    parser.add_argument(
        'mlmodel_output', nargs='?', type=str,
        help='a .mlmodel output file.'
    )

    args = parser.parse_args(argv)

    original_keras_model = keras.models.load_model(args.keras_checkpoint)
    img_size = original_keras_model.input_shape[1]
    num_classes = original_keras_model.output_shape[0][-1]

    keras_model = ICNetModelFactory.build(
        img_size,
        num_classes,
        alpha=args.alpha,
        weights_path=args.keras_checkpoint,
        train=False
    )

    mlmodel = coremltools.converters.keras.convert(
        keras_model,
        input_names='image',
        image_input_names='image',
        image_scale=2.0 / 255.0,
        red_bias=-1.0,
        green_bias=-1.0,
        blue_bias=-1.0,
        output_names='output'
    )

    mlmodel.save(args.mlmodel_output)


if __name__ == '__main__':
    convert(sys.argv[1:])
