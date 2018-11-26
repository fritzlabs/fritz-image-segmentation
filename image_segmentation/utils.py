import matplotlib.pyplot as pyplot
import numpy
import skimage.transform


def plot_image_and_mask(img, mask, alpha=0.6, deprocess_func=None,
                        reference_mask=None,
                        show_original_image=True,
                        small=False):
    """Plot an image and overlays a transparent segmentation mask.

    Args:
        img (arr): the image data to plot
        mask (arr): the segmentation mask
        alpha (float, optional): the alpha value of the segmentation mask.
        small: If true, output small figure

    Returns:
        pyplot.plot: a plot
    """
    max_mask = numpy.argmax(mask, axis=-1)

    rows, columns = 1, 1
    if show_original_image:
        columns += 1
    if reference_mask is not None:
        columns += 1

    fig = pyplot.figure()

    if deprocess_func:
        img = deprocess_func(img)

    # Add Results plot
    column_index = 1
    fig.add_subplot(rows, columns, column_index)

    pyplot.imshow(img.astype(int))
    pyplot.imshow(
        skimage.transform.resize(
            max_mask,
            img.shape[:2],
            order=0),
        alpha=alpha)

    if reference_mask is not None:
        column_index += 1
        fig.add_subplot(rows, columns, column_index)
        pyplot.imshow(img.astype(int))
        pyplot.imshow(
            skimage.transform.resize(
                reference_mask[:, :, 0],
                img.shape[:2],
                order=0),
            alpha=alpha)

    if show_original_image:
        column_index += 1
        fig.add_subplot(rows, columns, column_index)
        pyplot.imshow(img.astype('uint8'))

    if small:
        fig.set_size_inches(columns * 5, 5)
    else:
        fig.set_size_inches(columns * 10, 10)

    return fig


def plot_pixel_probabilities(probabilities, class_labels, subplot=None):
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
    total_items = num_classes + (1 if subplot else 0)
    columns = 4
    rows = numpy.ceil(total_items / 4)
    fig = pyplot.figure(figsize=(12, rows * 4))

    if subplot:
        fig.add_subplot(subplot)

    for cidx in range(num_classes):
        ax = fig.add_subplot(rows, columns, cidx + 1)
        ax.imshow(probabilities[:, :, cidx], vmin=0, vmax=1.0)
        ax.set_title(class_labels[cidx])
    fig.tight_layout()
    return fig
