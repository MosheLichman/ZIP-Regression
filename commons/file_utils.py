"""
File methods wrapper. For the loading wrappers it just prints out loading times.
For the saving wrappers, if creates the dir if it doesn't exist and takes care of the permissions.

Authors:
    1. Dimitrios Kotzias
    2. Moshe Lichman
"""
import numpy as np
import time
import os
from os.path import join

from commons import log_utils as log


def make_dir(path):
    """
    Making sure that the dir exist. If not, creating it with the write permissions.

     INPUT:
    -------
        1. path:    <string>    dir path
    """
    if not os.path.exists(path):
        os.makedirs(path)
        os.chmod(path, 0770)


def np_load(file_path):
    """
    Wrapper fpr the np.load that also prints time.

     INPUT:
    -------
        1. file_path:   <string>    file path

     OUTPUT:
    --------
        1. data:    <?>     whatever was saved

     RAISE:
    -------
        1. IOError
    """
    log.info('Loading %s' % file_path)
    start = time.time()
    data = np.load(file_path)
    log.info('Loading took %d seconds' % (time.time() - start))

    return data


def np_save(path, file_name, data):
    """
    Wrapper for np.save that also creates the dir if doesn't exist

     INPUT:
    -------
        1. path:        <sting>     dir path
        2. file_name:   <string>    file name
        3. data:        <ndarray>   numpy array
    """
    log.info('Saving file %s/%s' % (path, file_name))
    make_dir(path)

    start = time.time()
    np.save(join(path, file_name), data)
    os.chmod(join(path, file_name), 0770)
    log.info('Saving took %d seconds' % (time.time() - start))

