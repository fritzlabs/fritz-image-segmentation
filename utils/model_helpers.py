import tempfile
import os

import coremltools
import tensorflow as tf
import PIL.Image
import skimage.transform
import skimage.filters
import numpy
from tensorflow.python.platform import gfile
from image_segmentation import data_generator
import image_segmentation
import requests
from io import BytesIO


class ModelParameters(object):

    def __init__(self, **params):
        self.label_set = params['label_set']
        self.batch_size = params['batch_size']
        self.resolution = params['resolution']
        self.alpha = params['alpha']
        self.labels = params['labels']
        self.num_classes = len(self.labels)
        self.gcs_bucket = params.get('gcs_bucket')
        self._training_data_path = params.get('training_data_path')
        self._model_path = params.get('model_path')

        self.file_base = params.get(
            'file_base',
            f'{self.label_set}_{self.resolution}x{self.resolution}_1'
        )

    @property
    def training_data_path(self):
        if self._training_data_path:
            return self._training_data_path

        return (
            '../fritz-image-segmentation/data/'
            '{label_set}/{label_set}.tfrecord'
        ).format(label_set=self.label_set)

    @property
    def model_path(self):
        if self._model_path:
            return self._model_path

        return (
            f'gs://{self.gcs_bucket}/train/{self.file_base}.h5'
        )


class TrainedModel(object):

    def __init__(self, model_parameters):
        self._params = model_parameters
        resolution = model_parameters.resolution

        self.dataset = data_generator.ADE20KDatasetBuilder.build(
            self._params.training_data_path,
            self._params.batch_size,
            (resolution, resolution),
            self._params.num_classes,
            augment_images=False,
            repeat=False
        )

        self._model = None

    def download_and_build_model(self):
        temp_h5 = tempfile.NamedTemporaryFile(suffix='.h5')
        print("Loading model")
        # with gfile.Open(self._params.model_path, 'rb') as fid:
        #     temp_h5.file.write(fid.read())
        #     temp_h5.seek(0)

        return image_segmentation.icnet.ICNetModelFactory.build(
            self._params.resolution,
            self._params.num_classes,
            alpha=self._params.alpha,
            weights_path=self._params.model_path,
            train=False
        )

    @property
    def model(self):
        if self._model is None:
            self._model = self.download_and_build_model()

        return self._model

    def iterate_images(self):
        iterator = self.dataset.make_one_shot_iterator()
        el = iterator.get_next()

        try:
            with tf.Session() as sess:
                while True:
                    out = sess.run([el])
                    for i in range(out[0]['image'].shape[0]):
                        image = out[0]['image'][i]
                        mask = out[0]['mask'][i]
                        yield (image, mask)
        except tf.errors.OutOfRangeError:
            return

    def training_images(self, num_images=10, start_index=0):
        results = []
        for i, (image, mask) in enumerate(self.iterate_images()):
            if i < start_index:
                continue

            if len(results) >= num_images:
                break
            results.append((image, mask))

        return results

    def run_prediction(self, img_path=None, img_data=None, img_url=None,
                       img=None):
        if img_url:
            response = requests.get(img_url)
            img = PIL.Image.open(BytesIO(response.content))
        elif img_path:
            img = PIL.Image.open(img_path)

        if img_data is None:
            img = img.resize((self._params.resolution,
                              self._params.resolution))
            img_data = numpy.array(img)
            img_data = img_data * 1. / 255. - 0.5
            img_data = skimage.filters.gaussian(img_data, sigma=0.0)
        elif img_data is None:
            raise Exception("Must either pass image data or a path to image")

        return self.model.predict(img_data[None, :, :, :])

    def predict_and_plot(self, img_path=None, img_data=None, img_url=None,
                         mask=None, probabilities=True):
        if img_url:
            response = requests.get(img_url)
            img = PIL.Image.open(BytesIO(response.content))
            img = img.resize((self._params.resolution,
                              self._params.resolution))
        if img_path:
            img = PIL.Image.open(img_path)
            img = img.resize((self._params.resolution,
                              self._params.resolution))
        elif img_data is not None:
            img = ((img_data + 0.5) * 255).astype('uint8')

        output = self.run_prediction(img_path=img_path, img_data=img_data,
                                     img_url=img_url)

        figure = image_segmentation.utils.plot_image_and_mask(
            numpy.array(img),
            output[0],
            reference_mask=mask,
            alpha=0.9,
            small=True)
        generated_figures = [figure]

        if probabilities:
            generated_figures.append(
                image_segmentation.utils.plot_pixel_probabilities(
                    output[0],
                    self._params.labels
                )
            )

        return output, generated_figures

    def calculate_error(self, results, mask):
        resized_mask = numpy.resize(mask[:, :, 0], (
            results.shape[0], results.shape[1]
        ))
        resized_mask = mask[:, :, 0]

        resized_results = skimage.transform.resize(
            numpy.argmax(results, axis=-1),
            mask.shape[:2],
            preserve_range=True,
            anti_aliasing=False,
            order=0)

        diff = resized_mask - resized_results

        success_rate = []
        for i, label in enumerate(self._params.labels):
            total_class_values = numpy.sum(resized_mask == i)
            if not total_class_values:
                continue

            incorrect = float(numpy.count_nonzero(diff[resized_mask == i]))
            true_positive = float(numpy.sum(diff[resized_mask == i] == 0))
            iou = true_positive / (true_positive + incorrect)
            success_rate.append((i, total_class_values, true_positive, iou))
            print(f"{label} - total: {total_class_values}, IoU: {iou}")

        mean_iou = (
            sum([iou for _, _, _, iou in success_rate]) / len(success_rate)
        )
        print(f"mIoU: {mean_iou}")
        return success_rate

    def convert_to_coreml(self, mlmodel_path='./'):
        mlmodel = coremltools.converters.keras.convert(
            self.model,
            input_names='image',
            image_input_names='image',
            image_scale=1.0 / 255.0,
            red_bias=-0.5,
            green_bias=-0.5,
            blue_bias=-0.5,
            output_names='output'
        )
        mlmodel_file_path = (
            os.path.join(mlmodel_path, self._params.file_base + '.mlmodel')
        )
        mlmodel.save(mlmodel_file_path)
        print(f"successfully saved {mlmodel_file_path}")
