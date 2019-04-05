import numpy as np
from .utils import multinomial


def utility_for_consumers(
    quality,
    prev_usage,
    usage_counter,
    privacy_concern,
    firm_privacy_score,
    util_weight_dict,
):
    """
    Input:
    - quality = (firm, category)
    - prev_usage = (consumer, category, firm)
    - usage_counter = (consumer, category, firm)
    - privacy_concern = (consumer)
    - firm_privacy_score = (firm)
    - util_weight_dict has all the weights we need
    returns (Customer, Category, Firm): utility of each product for the customer (whether or not it exists).
    """
    n_consumers, n_categories, n_firms = prev_usage.shape
    usage_company = usage_counter.sum(axis=1)
    w_priv = util_weight_dict["w_priv"]
    w_qual = util_weight_dict["w_qual"]
    w_loyal_firm = util_weight_dict["w_loyal_firm"]
    w_loyal_category = util_weight_dict["w_loyal_category"]
    return (
        w_qual * quality.T[None, :, :]
        + w_loyal_category * usage_counter
        + w_loyal_firm * usage_company[:, None, :]
        - w_priv
        * privacy_concern[:, None, None]
        * (1 - firm_privacy_score)[None, None, :]
    )


def choose_firms(U, quality, w_logit, privacy_mask, rng):
    """
    input:
     - utility U: (consumer, category, firm) matrix of utilities
     - quality: (firm, category)
     - w_logit: constant
     - privacy_mask:
    output: (consumer, category, firm) one hot matrix of choice
    """
    market_matrix = (quality > 0).astype(int).T
    U_exp = np.exp(w_logit * U) * market_matrix[None, :, :] * privacy_mask[:, None, :]
    prob = np.nan_to_num(U_exp / np.nansum(U_exp, axis=-1, keepdims=True))
    choice = multinomial(prob, rng)
    return choice
