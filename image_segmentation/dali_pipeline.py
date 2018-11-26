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
                 index_path,
                 config,
                 shard_id=0):

        super(CommonPipeline, self).__init__(batch_size,
                                             num_threads,
                                             device_id)

        self.image_size = image_size
        self.input = self._input(tfrecord_path, index_path, shard_id=shard_id)
        # The nvjpeg decoder throws an error for some unsupported jpegs.
        # until this is fixed, we'll use the host decoder, which runs on the
        # CPU.
        # self.decode = ops.nvJPEGDecoder(device="mixed",
        #                                 output_type=types.RGB)
        self.decode = ops.HostDecoder(device="cpu",
                                      output_type=types.RGB)
        self.resize = ops.Resize(device="gpu",
                                 image_type=types.RGB,
                                 interp_type=types.INTERP_LINEAR,
                                 resize_x=image_size,
                                 resize_y=image_size)

        self.resize_large = ops.Resize(device="gpu",
                                       image_type=types.RGB,
                                       interp_type=types.INTERP_LINEAR,
                                       resize_x=image_size * config.zoom_scale,
                                       resize_y=image_size * config.zoom_scale)

        self.color_twist = ops.ColorTwist(
            device="gpu",
        )
        self.crop_mirror_normalize = ops.CropMirrorNormalize(
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
        self.flip = ops.Flip(device="gpu")

        self.coin = ops.CoinFlip(probability=0.5)
        self.rotate_rng = ops.Uniform(range=(config.rotate_angle_min,
                                             config.rotate_angle_max))
        self.crop_x_rng = ops.Uniform(range=(0.0, config.crop_x_max))
        self.crop_y_rng = ops.Uniform(range=(0.0, config.crop_y_max))
        self.hue_rng = ops.Uniform(range=(config.hue_min,
                                          config.hue_max))
        self.contrast_rng = ops.Uniform(range=(config.contrast_min,
                                               config.contrast_max))
        self.saturation_rng = ops.Uniform(range=(config.saturation_min,
                                                 config.saturation_max))
        self.brightness_rng = ops.Uniform(range=(config.brightness_min,
                                                 config.brightness_max))

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
        images = images.gpu()
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
        masks = masks.gpu()
        masks = self.resize_large(masks)
        masks = self.rotate(masks, angle=angle)
        masks = self.crop(masks, crop_pos_x=crop_x, crop_pos_y=crop_y)
        masks = self.resize(masks)
        masks = self.flip(masks, horizontal=coin)

        images = self.crop_mirror_normalize(images)
        masks = self.cast(masks)
        return (images, masks)

    def iter_setup(self):
        pass
