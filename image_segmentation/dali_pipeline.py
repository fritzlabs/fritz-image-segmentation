import tensorflow as tf
from nvidia import dali
import nvidia.dali.tfrecord as tfrec
from nvidia.dali import ops
from nvidia.dali import types
import nvidia.dali.plugin.tf as dali_tf
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy


class CommonPipeline(dali.pipeline.Pipeline):

    def _input(self, tfrecord_path, index_path):
        print(index_path)
        return ops.TFRecordReader(
            path=tfrecord_path,
            index_path=index_path,
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
                 index_path):

        super(CommonPipeline, self).__init__(batch_size, num_threads,
                                             device_id)

        self.image_size = image_size
        self.input = self._input(tfrecord_path, index_path)
        self.decode = ops.nvJPEGDecoder(device="mixed",
                                        output_type=types.RGB)
        self.host_decode = ops.HostDecoder(
            device="cpu",
            output_type=types.DALIImageType.RGB
        )
        self.resize = ops.Resize(device="gpu",
                                 image_type=types.RGB,
                                 interp_type=types.INTERP_LINEAR,
                                 resize_x=image_size,
                                 resize_y=image_size)
        mask_resizes = {}
        for scale in [4, 8, 16]:
            mask_resizes[scale] = ops.Resize(
                device="gpu",
                image_type=types.RGB,
                interp_type=types.INTERP_LINEAR,
                resize_x=image_size // scale,
                resize_y=image_size // scale,
            )

        self.mask_resizes = mask_resizes
        self.cmn = ops.CropMirrorNormalize(
            device="gpu",
            crop=image_size,
            output_dtype=types.FLOAT,
            image_type=types.RGB,
            output_layout=types.DALITensorLayout.NHWC,
            mean=122.5,
            std=255.0
        )
        self.coin = ops.CoinFlip(probability=0.5)
        self.coin2 = ops.CoinFlip(probability=0.5)

        self.rotate = ops.Rotate(
            device="gpu",
            fill_value=0
        )

        self.flip = ops.Flip(device="gpu")
        self.uniform = ops.Uniform(range=(0.0, 1.0))
        self.resize_rng = ops.Uniform(range=(256, 480))
        self.rotate_rng = ops.Uniform(range=(-20, 20))
        self.iter = 0

    def define_graph(self):
        inputs = self.input()
        angle = self.rotate_rng()
        images = self.decode(inputs["image/encoded"])
        images = self.resize(images)
        images = self.cmn(images)
        images = self.rotate(images, angle=angle)

        masks = self.host_decode(inputs["image/segmentation/class/encoded"])
        masks = masks.gpu()
        masks = self.resize(masks)
        masks = self.rotate(masks, angle=angle)

        coin = self.coin()
        coin2 = self.coin2()

        images = self.flip(images, horizontal=coin, vertical=coin2)
        masks = self.flip(masks, horizontal=coin, vertical=coin2)

        resized_masks = {}
        for scale in [4, 8, 16]:
            resize = self.mask_resizes[scale]
            resized_masks[scale] = resize(masks)

        return (images, resized_masks[4], resized_masks[8], resized_masks[16])

    def iter_setup(self):
        pass
