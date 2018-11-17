"""Setup script for image_segmentation."""

from setuptools import find_packages
from setuptools import setup


REQUIRED_PACKAGES = [
    'h5py',
    'keras==2.1.2',
    'Pillow',
    'matplotlib',
    'google-cloud-storage',
]

setup(
    name='image_segmentation',
    version='1.0',
    install_requires=REQUIRED_PACKAGES,
    include_package_data=True,
    packages=[
        p for p in find_packages()
        if p.startswith('image_segmentation') or p.startswith('utils')
    ],
    description='Fritz Style Image Segmentation Library',
)
