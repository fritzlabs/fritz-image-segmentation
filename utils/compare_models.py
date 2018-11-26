from matplotlib import gridspec
from matplotlib import pyplot
import skimage.transform
import numpy


def create_pascal_label_colormap():
    """Creates a label colormap used in PASCAL VOC segmentation benchmark.

    Returns:
        A Colormap for visualizing segmentation results.
    """
    colormap = numpy.zeros((256, 3), dtype=int)
    ind = numpy.arange(256, dtype=int)

    for shift in reversed(range(8)):
        for channel in range(3):
            colormap[:, channel] |= ((ind >> channel) & 1) << shift
        ind >>= 3

    return colormap


def label_to_color_image(label):
    """Adds color defined by the dataset colormap to the label.

    Args:
      label: A 2D array with integer type, storing the segmentation label.

    Returns:
      result: A 2D array with floating type. The element of the array
        is the color indexed by the corresponding element in the inumpyut label
        to the PASCAL color map.

    Raises:
      ValueError: If label is not of rank 2 or its value is larger than color
        map maximum entry.
    """
    if label.ndim != 2:
        raise ValueError('Expect 2-D inumpyut label')

    colormap = create_pascal_label_colormap()

    if numpy.max(label) >= len(colormap):
        raise ValueError('label value too large.')

    return colormap[label]


LABEL_NAMES = numpy.asarray([
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tv'
])

FULL_LABEL_MAP = numpy.arange(len(LABEL_NAMES)).reshape(len(LABEL_NAMES), 1)
FULL_COLOR_MAP = label_to_color_image(FULL_LABEL_MAP)


def vis_segmentation(image, deeplab_seg_map, icnet_seg_map):
    """Visualizes inumpyut image, segmentation map and overlay view."""
    pyplot.figure(figsize=(15, 5))
    grid_spec = gridspec.GridSpec(1, 4, width_ratios=[4, 4, 4, 4])

    pyplot.subplot(grid_spec[0])
    pyplot.imshow(image)
    pyplot.axis('off')
    pyplot.title('Input Image')

    pyplot.subplot(grid_spec[1])
    seg_image = label_to_color_image(deeplab_seg_map).astype(numpy.uint8)
    pyplot.imshow(seg_image)
    pyplot.axis('off')
    pyplot.title('Deeplab v3 Segmentation')

    pyplot.subplot(grid_spec[2])
    # resize icnet mask
    icnet_seg_map = skimage.transform.resize(
        icnet_seg_map[0, :, :],
        deeplab_seg_map.shape,
        preserve_range=True,
        anti_aliasing=False,
        order=0).astype('int')
    seg_image = label_to_color_image(icnet_seg_map).astype(numpy.uint8)
    pyplot.imshow(seg_image)
    pyplot.axis('off')
    pyplot.title('Fritz Segmentation')

    pyplot.subplot(grid_spec[3])
    pyplot.imshow(image)
    pyplot.imshow(seg_image, alpha=0.7)
    pyplot.axis('off')
    pyplot.title('Fritz Segmentation Overlay')

    pyplot.grid('off')
    pyplot.show()


def multiple_vis(results):

    fig = pyplot.figure(figsize=(15, 3 * len(results)))
    grid_spec = gridspec.GridSpec(len(results), 4, width_ratios=[4, 4, 4, 4])

    i = 0
    for image, deeplab_seg_map, icnet_seg_map in results:
        pyplot.subplot(grid_spec[i])
        pyplot.imshow(image)
        # pyplot.axis('off')
        i += 1

        pyplot.subplot(grid_spec[i])
        seg_image = label_to_color_image(deeplab_seg_map).astype(numpy.uint8)
        pyplot.imshow(seg_image)
        pyplot.axis('off')
        pyplot.title('Deeplab v3 Segmentation')

        i += 1
        pyplot.subplot(grid_spec[i])
        # resize icnet mask
        icnet_seg_map = skimage.transform.resize(
            icnet_seg_map[0, :, :],
            deeplab_seg_map.shape,
            preserve_range=True,
            anti_aliasing=False,
            order=0).astype('int')
        seg_image = label_to_color_image(icnet_seg_map).astype(numpy.uint8)
        pyplot.imshow(seg_image)
        pyplot.axis('off')
        pyplot.title('Fritz Segmentation')
        i += 1

        pyplot.subplot(grid_spec[i])
        pyplot.imshow(image)
        pyplot.imshow(seg_image, alpha=0.7)
        pyplot.axis('off')
        pyplot.title('Fritz Segmentation Overlay')
        i += 1

    pyplot.grid('off')

    return fig
