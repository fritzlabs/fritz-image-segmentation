import matplotlib.pyplot as pyplot
import numpy
import skimage.transform


def plot_image_and_mask(img, mask, alpha=0.6):
    """Plot an image and overlays a transparent segmentation mask.

    Args:
        img (arr): the image data to plot
        mask (arr): the segmentation mask
        alpha (float, optional): the alpha value of the segmentation mask.

    Returns:
        pyplot.plot: a plot
    """
    max_mask = numpy.argmax(mask, axis=-1)
    pyplot.imshow((img * 255 + 127.5).astype(int))
    pyplot.imshow(
        skimage.transform.resize(
            max_mask,
            img.shape[:2],
            order=0),
        alpha=alpha)
    pyplot.gcf().set_size_inches(10, 10)
    return pyplot.gcf()


def plot_pixel_probabilities(probabilities, class_labels):
    """Plot probabilities that each pixel belows to a given class.

    This creates a subplot for each class and plots a heatmap of
    probabilities that each pixel belongs to each class.

    Args:
        probabilities (arr): an array of class probabilities for each pixel
        class_labels (List[str]): the labels for each class

    Returns:
        TYPE: Description
    """
    num_classes = probabilities.shape[-1]
    columns = 4
    rows = numpy.ceil(num_classes / 4)
    fig = pyplot.figure(figsize=(12, rows * 4))
    for cidx in range(num_classes):
        ax = fig.add_subplot(rows, columns, cidx + 1)
        ax.imshow(probabilities[:, :, cidx], vmin=0, vmax=1.0)
        ax.set_title(class_labels[cidx])
    fig.tight_layout()
    return fig
