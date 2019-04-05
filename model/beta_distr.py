"""
Helpful stuff re beta distributions
characterized by mode and variance
"""

from scipy import optimize


def sd_fun(var, m):
    """
    Gives a function f such that f(alpha) = 0
    m = mode of beta function
    var = variance of beta function
    """

    def fn(x):
        # express beta in terms of alpha (x) and the mode
        b = x / m * (1 - m) + 2 - 1 / m
        return x * b - var * (x + b) ** 2 * (x + b + 1)

    return fn


def get_beta_params(m, var):
    """
    finds parameters alpha (a), beta (b) of a beta function with given
    m = mode
    var = variance
    """
    a = optimize.brenth(sd_fun(var, m), 1, 1000)
    b = a * (1 - m) / m - 1 / m + 2
    return a, b
