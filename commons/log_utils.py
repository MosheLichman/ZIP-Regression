"""
My version of logger that wraps the python default logger. It gives simpler controllers to the things I need like print
to stderr (or out) and to files in my format (I don't need the rest).

Author: Moshe Lichman
"""
import logging
import sys

_my_format = '%(asctime)s %(levelname)s --> %(message)s'

"""
By default the logger logs to the stderr on info.
"""

_logger = logging.getLogger('my_logger')
_logger.setLevel(logging.DEBUG)    # The logger should be set to the lowest level.

_stderr_hd = logging.StreamHandler(sys.stderr)
_stderr_hd.setLevel(logging.INFO)
_stderr_hd.setFormatter(logging.Formatter(_my_format))

_logger.handlers = [_stderr_hd]


def log_to_file(file_path, level=10):
    """Adds the file as a logging target -- appends to it if one already exists. """
    info('LOGGER.log_to_file: Adding logging to file %s' % file_path)

    _file_hd = logging.FileHandler(file_path)
    _file_hd.setLevel(level)
    _file_hd.setFormatter(logging.Formatter(_my_format))

    _logger.handlers = [_file_hd, _stderr_hd]


def reset_logger():
    """Removes all handlers and keep just stderr one. """
    _logger.handlers = [_stderr_hd]


def set_verbose():
    """Sets log level to DEBUG. """
    _stderr_hd.setLevel(logging.DEBUG)


def no_logging():
    """Remove all handlers including the stderr one. """
    _logger.handlers = []

def debug(message):
    _logger.debug(message)


def info(message):
    _logger.info(message)


def error(message):
    _logger.error(message)


def warn(message):
    _logger.warning(message)
