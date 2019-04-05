"""
Things to do with the needs of consumers for products
"""

import numpy as np
from scipy.stats import beta

from .beta_distr import get_beta_params
from .utility import multinomial


def discretize_a_composite_beta(modes, vars, n_bins=500):
    """
    Computes the weight of a composite beta function with given modes and variances
    on n_bins on the unit interval
    """
    xs = np.linspace(0, 1, n_bins + 1)
    cdf_vals = np.zeros(n_bins)
    # loop over all beta distributions to be included
    for m, v in zip(modes, vars):
        a, b = get_beta_params(m, v)
        # get cdf in all points
        cdf = beta.cdf(xs, a, b)
        # diff cdf to get integral of pdf in that interval
        cdf_diff = cdf[1:] - cdf[:-1]
        cdf_vals += cdf_diff
    return cdf_vals / len(modes)


def draw_from_one_need_distribution(modes, vars, n_choices, rng, n_bins=500):
    """
    Draws n_choices according to a mix of beta distributions described by modes and varsself.
    Uses the discretised version of this mix.
    """
    assert len(modes) == len(vars), "modes and vars of different lengths"
    discretized_beta = discretize_a_composite_beta(modes, vars, n_bins)
    choice = multinomial(discretized_beta[None, :] * np.ones(n_choices)[:, None], rng)
    bin_choices = np.where(choice)[1]
    xs = np.linspace(0, 1, n_bins + 1)
    return xs[bin_choices]
