import torch

from src.mmd.signature_functions import get_level_k_signatures_from_paths


def lambda_k(X, Y, k):
    """
    Calculates the level-k signature combination

                        \Lambda_k(X,Y) = \sum_{i \in I_k} ES(X)^{i} ES(Y)^{i}

    where I_k = \{1, \dots, d\}^k, and d is the dimension of the paths X, Y, and ES is the expected signature.

    :param X:   Path bank, N x l x d
    :param Y:   Path bank, N x l x d
    :param k:   Order to calculate up to
    :return:    \Lambda_k(X, Y)
    """

    if k == 0:
        return 1.

    sigX = torch.mean(get_level_k_signatures_from_paths(X, k), dim=0)
    sigY = torch.mean(get_level_k_signatures_from_paths(Y, k), dim=0)

    return torch.dot(sigX, sigY)


def gramda_k(X, Y, k):
    """
    Calculates the Gram matrix associated to two sets of paths X, Y at level k.

    :param X:   Path bank, N x l x d
    :param Y:   Path bank, N x l x d
    :param k:   Order of signature
    :return:    Gram matrix at level k
    """

    sigsX = get_level_k_signatures_from_paths(X, k)
    sigsY = get_level_k_signatures_from_paths(Y, k)

    return torch.einsum("ik,jk->ij", sigsX, sigsY)


def level_k_contribution(X, Y, k, phik=lambda x: 1, unbiased=True):
    """
    Calculates the level-k contribution associated to the paths X, Y

    :param X:           Path bank, N x l x d
    :param Y:           Path bank, N x l x d
    :param k:           Order of signature
    :param phik:        Scaling function
    :param unbiased:    Whether to estimate unbiased or biased statistic.
    :return:            M_k as written in the paper
    """
    if k == 0:
        return 0.

    N = X.size(0)

    gXX = gramda_k(X, X, k)
    gYY = gramda_k(Y, Y, k)
    gXY = gramda_k(X, Y, k)

    if unbiased:
        gXX -= torch.diag(torch.diag(gXX))
        gYY -= torch.diag(torch.diag(gYY))
        denom = N*(N-1)
    else:
        denom = N*N

    res1 = (torch.sum(gXX) + torch.sum(gYY)) / denom
    res2 = 2 * torch.sum(gXY) / (N ** 2)

    return phik(k) * (res1 - res2)


def kernel_est_k(X, Y, N, phik=lambda x: 1):
    """
    Calculates the estimated (truncated) kernel k^N_{\text{sig}} up to the order N

    :param X:       Path bank, N x l x d
    :param Y:       Path bank, N x l x d
    :param N:       Order of signature to calculate up to
    :param phik:    Signature scaling function
    :return:        k^N_\text{sig}(X, Y)
    """
    return torch.sum(torch.tensor([phik(_ki) * lambda_k(X, Y, _ki) for _ki in range(N+1)]))


def mmd_est_k(X, Y, N, phik=lambda x: 1, unbiased=True):
    """
    Calculates the truncated MMD up to order N between paths X, Y

    :param X:           Path bank, N x l x d
    :param Y:           Path bank, N x l x d
    :param N:           Order of signature to calculate up to
    :param phik:        Signature scaling function
    :param unbiased:    Whether to calculate unbiased or biased test statistic
    :return:            \mathcal{D}^N_\text{sig}(X, Y)
    """
    return torch.sum(torch.tensor([level_k_contribution(X, Y, ki, phik, unbiased) for ki in range(N+1)]), dtype=torch.float64)
