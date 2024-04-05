import matplotlib.pyplot as plt
import numpy as np


def golden_dimensions(width: float) -> tuple:
    """
    Returns a tuple of l x w in the golden ratio

    :param width:   Width parameter
    :return:        Tuple of l x w in golden ratio
    """

    gr = (1+np.sqrt(5))/2

    return gr*width, width


def make_grid(axis=None):
    _plt_obj = axis if axis is not None else plt
    getattr(_plt_obj, "grid")(visible=True, color='grey', linestyle=':', linewidth=1.0, alpha=0.3)
    getattr(_plt_obj, "minorticks_on")()
    getattr(_plt_obj, "grid")(visible=True, which='minor', color='grey', linestyle=':', linewidth=1.0, alpha=0.1)
