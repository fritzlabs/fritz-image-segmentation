import logging

logger = logging.getLogger(__name__)


def build_config(**updates):
    defaults = {
        'hue_min': -30,
        'hue_max': 30,
        'zoom_scale': 1.3,
        'rotate_angle_min': -45,
        'rotate_angle_max': 45,
        'crop_x_max': 0.2,
        'crop_y_max': 0.2,
        'contrast_min': 0.45,
        'contrast_max': 1.5,
        'saturation_min': 0.4,
        'saturation_max': 2.0,
        'brightness_min': 0.35,
        'brightness_max': 1.5,
    }
    for key in updates:
        if key not in defaults:
            raise Exception("Augmentation Config %s not found." % key)

    defaults.update(**updates)

    return defaults


class DaliConfig(object):
    """Wrapper for Dali augmentation yaml config parameters. """
    def __init__(self, **updates):

        self.__dict__ = build_config(**updates)

    def summarize(self):
        logger.info('Dali Image Augmentation Parameters')
        logger.info('==================================')
        for key, value in self.__dict__.items():
            logger.info('  %s: %s', key, value)
