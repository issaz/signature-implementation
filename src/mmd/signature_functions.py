import torch
import iisignature
import numpy as np


def get_signatures(paths, k):
    """
    Calculates signatures up to order k

    :param paths:   Bank of paths to
    :param k:       Order
    :return:        Bank of signatures
    """

    n, l, d = paths.shape

    if k == 0:
        return torch.zeros(n) + 1.

    if type(paths) == torch.Tensor:
        paths = np.array(paths.cpu().detach())

    sigs = iisignature.sig(paths, k)
    return torch.tensor(sigs)


def get_level_k_signatures_from_paths(paths, k):
    """
    Explicitly returns the level k signatures from the more general full signature.

    :param paths:       Bank of paths
    :param k:           Order of signature
    :return:            Level k terms
    """
    _, _, d = paths.shape

    sigs  = get_signatures(paths, k)
    stloc = sum((d ** ki for ki in range(k))) - 1

    return sigs[:, stloc:]


def get_level_k_signatures_from_signatures(signatures, k, d):
    """
    Extracts the level k terms from the bank of signatures directly. Requires more data in order to get the correct
    indexing

    :param signatures:      Bank of signatures
    :param k:               Order of level terms to extract
    :param d:               Dimension of path
    :return:                Level k signature terms
    """

    stloc = sum((d ** ki for ki in range(k))) - 1
    endloc = sum((d ** ki for ki in range(k+1))) - 1

    return signatures[:, stloc:endloc]
