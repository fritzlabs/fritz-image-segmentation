"""Setup script for image_segmentation."""

import logging
import subprocess
from setuptools import find_packages
from setuptools import setup
from setuptools.command.install import install


REQUIRED_PACKAGES = [
    'h5py',
    'keras==2.2.4',
    'Pillow',
    'matplotlib',
    'google-cloud-storage',
]


class CustomCommands(install):
    """A setuptools Command class able to run arbitrary commands."""

    def run_custom_command(self, command_list):
        p = subprocess.Popen(
            command_list,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT)
        # Can use communicate(input='y\n'.encode()) if the command run requires
        # some confirmation.
        stdout_data, _ = p.communicate()
        logging.info('Log command output: %s', stdout_data)
        if p.returncode != 0:
            raise RuntimeError('Command %s failed: exit code: %s' %
                               (command_list, p.returncode))

    def run(self):
        self.run_custom_command(['apt-get', 'update'])
        self.run_custom_command([
            'apt-get', 'install', '-y', 'python-tk'
        ])
        install.run(self)


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
    cmdclass={
        'install': CustomCommands,
    }
)
