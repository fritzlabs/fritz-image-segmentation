from nvidia import dali
import nvidia.dali.tfrecord as tfrec
from nvidia.dali import ops
from nvidia.dali import types


class CommonPipeline(dali.pipeline.Pipeline):

    def _input(self, tfrecord_path, index_path, shard_id=0):
        return ops.TFRecordReader(
            path=tfrecord_path,
            index_path=index_path,
            random_shuffle=True,
            features={
                'image/encoded': tfrec.FixedLenFeature((), tfrec.string, ""),
                'image/filename': tfrec.FixedLenFeature((), tfrec.string, ""),
                'image/format': tfrec.FixedLenFeature((), tfrec.string, ""),
                'image/height': tfrec.FixedLenFeature([1], tfrec.int64, -1),
                'image/width': tfrec.FixedLenFeature([1], tfrec.int64, -1),
                'image/channels': tfrec.FixedLenFeature([1], tfrec.int64, -1),
                'image/segmentation/class/encoded': (
                    tfrec.FixedLenFeature((), tfrec.string, "")
                ),
                'image/segmentation/class/format': (
                    tfrec.FixedLenFeature((), tfrec.string, "")
                )
            }
        )

    def __init__(self,
                 batch_size,
                 num_threads,
                 device_id,
                 image_size,
                 tfrecord_path,
                 index_path, shard_id=0):

        super(CommonPipeline, self).__init__(batch_size, num_threads,
                                             device_id)

        self.image_size = image_size
        self.input = self._input(tfrecord_path, index_path, shard_id=shard_id)
        self.decode = ops.nvJPEGDecoder(device="mixed",
                                        output_type=types.RGB)
        self.resize = ops.Resize(device="gpu",
                                 image_type=types.RGB,
                                 interp_type=types.INTERP_LINEAR,
                                 resize_x=image_size,
                                 resize_y=image_size)

        self.resize_large = ops.Resize(device="gpu",
                                       image_type=types.RGB,
                                       interp_type=types.INTERP_LINEAR,
                                       resize_x=image_size * 1.3,
                                       resize_y=image_size * 1.3)

        self.color_twist = ops.ColorTwist(
            device="gpu",
        )
        self.hue_rng = ops.Uniform(range=(-30, 30))
        self.contrast_rng = ops.Uniform(range=(0.45, 1.5))
        self.saturation_rng = ops.Uniform(range=(0.4, 2.0))
        self.brightness_rng = ops.Uniform(range=(0.35, 1.5))

        self.cmn = ops.CropMirrorNormalize(
            device="gpu",
            crop=image_size,
            output_dtype=types.FLOAT,
            image_type=types.RGB,
            output_layout=types.DALITensorLayout.NHWC,
            mean=122.5,
            std=255.0
        )

        self.crop = ops.Crop(
            device="gpu",
            crop=image_size,
        )

        self.cast = ops.Cast(
            device="gpu",
            dtype=types.DALIDataType.INT64
        )
        self.rotate = ops.Rotate(
            device="gpu",
            fill_value=0
        )

        self.coin = ops.CoinFlip(probability=0.5)
        self.coin2 = ops.CoinFlip(probability=0.5)

        self.flip = ops.Flip(device="gpu")
        self.rotate_rng = ops.Uniform(range=(-45, 45))
        self.crop_x_rng = ops.Uniform(range=(0.0, 0.2))
        self.crop_y_rng = ops.Uniform(range=(0.0, 0.2))

        self.iter = 0

    def define_graph(self):
        inputs = self.input()
        angle = self.rotate_rng()
        coin = self.coin()
        hue = self.hue_rng()
        contrast = self.contrast_rng()
        saturation = self.saturation_rng()
        brightness = self.brightness_rng()
        crop_x = self.crop_x_rng()
        crop_y = self.crop_y_rng()

        images = self.decode(inputs["image/encoded"])
        images = self.resize_large(images)
        images = self.rotate(images, angle=angle)
        images = self.crop(images, crop_pos_x=crop_x, crop_pos_y=crop_y)
        images = self.resize(images)
        images = self.color_twist(images,
                                  brightness=brightness,
                                  hue=hue,
                                  saturation=saturation,
                                  contrast=contrast)
        images = self.flip(images, horizontal=coin)

        masks = self.decode(inputs["image/segmentation/class/encoded"])
        masks = self.resize_large(masks)
        masks = self.rotate(masks, angle=angle)
        masks = self.crop(masks, crop_pos_x=crop_x, crop_pos_y=crop_y)
        masks = self.resize(masks)
        masks = self.flip(masks, horizontal=coin)

        images = self.cmn(images)
        masks = self.cast(masks)
        return (images, masks)

    def iter_setup(self):
        pass
