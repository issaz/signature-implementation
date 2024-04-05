import torch

from tqdm import tqdm


def return_mmd_distributions(h0_paths, h1_paths, mmd, n_atoms=128, batch_size=32, verbose=True):
    """
    Returns null and alternate MMD distributions for a given kernel

    :param h0_paths:    Bank of paths under the null hypothesis
    :param h1_paths:    Bank of paths under the alternate hypothesis
    :param mmd:         Method to calculate the MMD
    :param n_atoms:     Number of atoms in the corresponding distributions
    :param batch_size:  Number of paths to sample from each distributiom
    :param verbose:     Whether to plot progress bar
    :return:
    """

    assert h0_paths.shape[0] == h1_paths.shape[0]

    path_bank_size = h0_paths.shape[0]

    h0_dists = torch.zeros(n_atoms)
    h1_dists = torch.zeros(n_atoms)

    rand_ints = torch.randint(0, path_bank_size, size=(n_atoms, 3, batch_size))

    itr = tqdm(rand_ints) if verbose else rand_ints

    with torch.no_grad():
        for i, ii in enumerate(itr):
            x1, x2 = h0_paths[ii[0]], h0_paths[ii[1]]
            y = h1_paths[ii[-1]]

            h0_dists[i] = mmd(x1, x2)
            h1_dists[i] = mmd(x1, y)

    return h0_dists, h1_dists


def expected_type2_error(h1_dist: torch.Tensor, crit_value: float):
    """
    Calculates the expected type II error given a critical value from a null distribution and the alternate distribution

    :param h1_dist:     MMD distribution under the alternate hypothesis
    :param crit_value:  Critical value associated to the null distribution
    :return:
    """
    n_atoms = h1_dist.shape[0]
    num_fail = h1_dist <= crit_value
    return sum(num_fail.type(torch.float32))/n_atoms
