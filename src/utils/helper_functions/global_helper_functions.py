import os
from pathlib import Path


def get_project_root() -> Path:
    """
    Returns path root of project

    :return: Path object
    """
    return get_source_root().parent


def get_source_root() -> Path:
    """
    Returns /src root of project

    :return: Path object
    """
    return Path(__file__).parent.parent.parent


def mkdir(filepath):
    """
    Makes a directory corresponding to :param filepath: if it does not exist.

    :param filepath:    Filepath to make directory of.
    :return:            None
    """
    if not os.path.exists(filepath):
        os.makedirs(filepath)
