import numpy as np
from .utils import min_max_scaler, multinomial


def F(x, alpha=10):
    """
    A square root function along which qualities of products move
    """
    return np.sqrt(x / alpha)


def F_inverse(x, alpha=10):
    """
    The inverse of F
    """
    return alpha * np.power(x, 2)


def investment_scaler(x, low=0.1, high=0.7, alpha=0.00001):
    """
    Use this on the investment to get the probability of investment success
    """
    return low + (high - low) * (1 - np.exp(-alpha * x))


def product_quality_update(quality_start, investment, alpha):
    """
    returns the additional quality gained from the investment.
    calculated as if everything was innovated, but there is a mask in the simulation
    """
    return (
        F(F_inverse(quality_start, alpha) + investment[:, None], alpha) - quality_start
    )


def enter_new_category(
    quality,
    category_datatype,
    firm_innovation_preference,
    category_total_usage,
    category_ticks_alive,
    innovation_dict,
    qual_diff,
    rng,
    n=None,
):
    """
    innovation into a category that doesn't yet exist in the company
    could be a new or existing category
    quality: (firm, category)
    category_datatype: (category, datatype)
    firm_innovation_preference: (firm, [existing category, new category]): one 0, one 1 per row
    """
    w_mean_usage = innovation_dict["w_mean_usage"]
    w_datatype = innovation_dict["w_datatype"]
    # establish for which categories the firm already has some datatypes
    firm_existing_cats = (quality > 0).astype(int)
    existing_cats = firm_existing_cats.sum(axis=0) > 0
    # a type of vialibility for categories
    cat_ever_existed = category_ticks_alive > 0
    mean_usage_per_tick = min_max_scaler(
        category_total_usage / np.maximum(1, category_ticks_alive)
    )
    # give some usage per tick for categories that don't yet exist
    mean_usage_per_tick[~cat_ever_existed] = np.median(
        mean_usage_per_tick[existing_cats]
    )
    # maximum quality per existing category
    max_quals = np.max(quality, axis=0)
    # for newly born firms entering existing categories
    if n:
        cat_util = np.exp(
            w_mean_usage * mean_usage_per_tick[None, :] * np.ones(n)[:, None]
        )
        cat_util *= existing_cats.astype(int)
        cat_prob = cat_util / np.sum(cat_util, axis=-1, keepdims=True)
    # for existing firms, entering existing or new categories
    else:
        # (Firm, Datatype): number of datatypes that firm already has in its portfolio
        existing_datatypes = quality.dot(category_datatype)
        # (Firm, Category): number of datatypes company already has
        cat_datatypes = existing_datatypes.dot(category_datatype.T)
        # set to zero for categories that the company already has
        cat_datatypes = min_max_scaler(cat_datatypes, axis=-1)
        # utilities
        cat_util = np.exp(
            w_datatype * cat_datatypes + w_mean_usage * mean_usage_per_tick[None, :]
        )
        # don't innovate in existing products
        cat_util[quality > 0] = 0
        # mask out existing categories for firms that want to innovate in new category
        cat_util[np.ix_(firm_innovation_preference[:, 1] == 1, existing_cats)] = 0
        # mask out non-existing categories for firms that want to innovatin in existing category
        cat_util[np.ix_(firm_innovation_preference[:, 0] == 1, ~existing_cats)] = 0
    cat_prob = cat_util / np.sum(cat_util, axis=-1, keepdims=True)
    chosen_cat = multinomial(cat_prob, rng)
    if not n:
        assert (chosen_cat * quality == 0).all, np.sum(chosen_cat * quality, axis=-1)
    # line below deals with firms that have died
    chosen_cat[chosen_cat.sum(axis=-1) == 0, 0] = 1
    choices = np.where(chosen_cat)[1]
    # establish the quality to be assigned
    max_qual_choices = max_quals[choices]
    # line below ensure that quality in a new category is set to one
    new_qual = np.maximum(
        max_qual_choices - rng.uniform(size=cat_prob.shape[0]) * qual_diff, 1
    )
    # return the new quality the product would have if innovation is successful
    return new_qual[:, None] * chosen_cat


def invest_utility_existing(quality, category_total_usage, tick, innovation_dict):
    """
    calculates, for all firms, the probability of investing in their existing products
    """
    w_num_firms_per_cat = innovation_dict["w_num_firms_per_cat"]
    w_usage = innovation_dict["w_usage"]
    existing_products = (quality > 0).astype(int)
    num_firms_per_cat = min_max_scaler(existing_products.sum(axis=0))
    usage_boost = min_max_scaler(
        category_total_usage[None, :] * existing_products, axis=1
    )

    firm_utility = (
        np.exp(w_num_firms_per_cat * num_firms_per_cat[None, :] + w_usage * usage_boost)
        * existing_products
    )
    return np.nan_to_num(firm_utility / firm_utility.sum(axis=-1, keepdims=True))


def apply_data_skill(invest_data_value_base, data_combination_skill):
    # Only apply data combination skill if there is more than one data type involved
    applicable = ((invest_data_value_base > 0).astype(int).sum(axis=1) > 1).astype(int)
    dcs = data_combination_skill * applicable
    return invest_data_value_base.sum(axis=-1) * (1 + dcs)
