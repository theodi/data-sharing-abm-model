import numpy as np

from .needs import draw_from_one_need_distribution


def setup_simulation(
    general_dict,
    seed_dict,
    rng,
    capital_dict,
    needs_dict,
    data_dict,
    category_dict,
    privacy_dict,
    openness_dict,
    innovation_dict,
):

    # grab constants
    n_ticks = general_dict["n_ticks"]
    n_init_firms = general_dict["n_init_firms"]
    n_init_big_firms = general_dict["n_init_big_firms"]
    n_consumers = general_dict["n_consumers"]
    n_total_categories = category_dict["n_total_categories"]
    n_init_categories = category_dict["n_init_categories"]

    # Estimating how many firms there will eventually be in the model (high upper bound)
    mean_new_firms = general_dict["birth_lambda"] * n_ticks
    var_new_firms = mean_new_firms
    max_new_firms = np.floor(mean_new_firms + 1 * np.sqrt(var_new_firms)).astype(int)
    n_total_firms = n_init_firms + max_new_firms

    # indicator vector: is the firm still in business?
    F_alive = np.concatenate([np.ones(n_init_firms), np.zeros(max_new_firms)])

    # assign capital to each firm
    capital = np.concatenate(
        [
            np.ones(n_init_big_firms) * capital_dict["big"],
            np.ones(n_init_firms - n_init_big_firms) * capital_dict["small"],
            np.zeros(max_new_firms),
        ]
    )
    # setting up usage matrix (consumer, category, firm)
    usage = np.zeros((n_consumers, n_total_categories, n_total_firms))

    # quality matrix set-up: zero if product not in firm portfolio
    quality = np.zeros((n_init_firms + max_new_firms, n_total_categories))
    # assigning one product to each company
    quality[
        np.arange(n_init_firms), np.mod(np.arange(n_init_firms), n_init_categories)
    ] = 1
    # big firms have higher quality products
    quality[:n_init_big_firms] *= 2

    # privacy score setup
    privacy_rng = np.random.RandomState(seed=seed_dict["privacy_seed"])
    firm_privacy_score = np.maximum(
        np.minimum(
            privacy_rng.normal(
                privacy_dict["mean_firm_score"],
                privacy_dict["var_firm_score"],
                size=n_init_firms + max_new_firms,
            ),
            1,
        ),
        0,
    )

    # consumer characteristics
    consumer_privacy_concern = np.maximum(
        np.minimum(
            privacy_rng.normal(
                privacy_dict["mean_cons_concern"],
                privacy_dict["var_cons_concern"],
                size=n_consumers,
            ),
            1,
        ),
        0,
    )
    consumer_wealth = 1 + rng.uniform(size=n_consumers) * 9

    # firm investment profile
    firm_investment_profile = np.zeros((n_init_firms + max_new_firms, 3))
    firm_investment_profile[:, 0] = innovation_dict["investment_profile"][
        "existing_product"
    ]
    firm_investment_profile[:, 1] = innovation_dict["investment_profile"]["new_product"]
    firm_investment_profile[:, 2] = innovation_dict["investment_profile"][
        "new_category"
    ]

    # Assign needs for categories
    need_matrix = np.zeros(shape=(n_consumers, n_total_categories))
    need_rng = np.random.RandomState(seed=seed_dict["need_seed"])
    n_modes_probs = needs_dict["n_modes_probs"]
    for i in range(np.minimum(n_init_big_firms, n_total_categories)):
        # assure the big firms are in high-need categories
        modality = need_rng.choice(np.arange(len(n_modes_probs)), p=n_modes_probs)
        if needs_dict["hyper_mode"] == "uniform":
            modes = need_rng.uniform(
                np.maximum(needs_dict["needs_range_mode_low"], 0.5),
                needs_dict["needs_range_mode_high"],
                modality,
            )
        if needs_dict["hyper_var"] == "uniform":
            vars = need_rng.uniform(
                needs_dict["needs_range_var_low"],
                needs_dict["needs_range_var_high"],
                modality,
            )
        need_matrix[:, i] = draw_from_one_need_distribution(
            modes, vars, n_consumers, need_rng
        )
    for i in range(
        np.minimum(n_init_big_firms, n_total_categories), n_total_categories
    ):
        modality = need_rng.choice(np.arange(len(n_modes_probs)), p=n_modes_probs)
        if needs_dict["hyper_mode"] == "uniform":
            modes = need_rng.uniform(
                needs_dict["needs_range_mode_low"],
                needs_dict["needs_range_mode_high"],
                modality,
            )
        if needs_dict["hyper_var"] == "uniform":
            vars = need_rng.uniform(
                needs_dict["needs_range_var_low"],
                needs_dict["needs_range_var_high"],
                modality,
            )
        need_matrix[:, i] = draw_from_one_need_distribution(
            modes, vars, n_consumers, need_rng
        )

    # datatypes for categories
    n_data_types_init = data_dict["n_data_types_init"]
    n_data_types_total = data_dict["n_data_types_total"]
    data_rng = np.random.RandomState(seed=seed_dict["data_seed"])
    category_datatype = np.zeros(
        (n_total_categories, n_data_types_total), dtype=np.int8
    )
    for j in range(n_init_categories):
        num_types = data_rng.choice(np.arange(np.minimum(n_data_types_init, 3))) + 1
        choice_types = data_rng.choice(
            np.arange(n_data_types_init), size=num_types, replace=False
        )
        category_datatype[j, choice_types] = 1
    for e, i in enumerate(range(n_init_categories, n_total_categories)):
        shift_num_types = np.floor(e / data_dict["growth_factor"]).astype(int)
        num_types = (
            data_rng.choice(
                np.arange(np.minimum(n_data_types_total, 3 + shift_num_types))
            )
            + 1
        )
        choice_types = data_rng.choice(
            np.arange(
                np.minimum(n_data_types_total, n_data_types_init + 3 * shift_num_types)
            ),
            size=num_types,
            replace=False,
        )
        category_datatype[i, choice_types] = 1
    # remove datatypes that are not used at all
    category_datatype = category_datatype[:, category_datatype.sum(axis=0) > 0]
    n_datatypes = category_datatype.shape[1]

    # usage counter, will be used to keep track of consumption - which will be discounted over time
    usage_counter = np.zeros((n_consumers, n_total_categories, n_total_firms))

    # tracker of usage, capital to decide on firm death
    ticks_no_usage = np.zeros((n_init_firms + max_new_firms))
    ticks_no_capital = np.zeros((n_init_firms + max_new_firms))

    # Data combination skills
    if data_dict["data_skill_distr"] == "uniform":
        low, hi = data_dict["data_skill_range_low"], data_dict["data_skill_range_high"]
        data_combination_skill = data_rng.choice(
            np.arange(low, hi), size=n_total_firms, replace=True
        )

    # Data portability
    # mask - firms that haven't turned down each others requests yet
    requestable = np.ones(
        (
            n_total_firms,
            n_total_categories,
            n_total_firms,
            n_total_categories,
            n_datatypes,
        ),
        dtype=np.int8,
    )
    # firms won't request their own data to be shared
    requestable[np.arange(n_total_firms), :, np.arange(n_total_firms)] = 0
    # only requestable if categories have the right datatype
    requestable = (
        requestable
        * category_datatype[None, :, None, None, :]
        * category_datatype[None, None, None, :, :]
    )
    # if there is a cartel, only get big firms will share
    if openness_dict["cartel"]:
        requestable[n_init_big_firms:] = 0
        requestable[:, :, n_init_big_firms:] = 0

    return {
        "capital": capital,
        "usage": usage,
        "quality": quality,
        "firm_privacy_score": firm_privacy_score,
        "consumer_privacy_concern": consumer_privacy_concern,
        "consumer_wealth": consumer_wealth,
        "F_alive": F_alive,
        "ticks_no_usage": ticks_no_usage,
        "ticks_no_capital": ticks_no_capital,
        "need_matrix": need_matrix,
        "category_datatype": category_datatype,
        "firm_investment_profile": firm_investment_profile,
        "usage_counter": usage_counter,
        "data_combination_skill": data_combination_skill,
        "requestable": requestable,
        "portability_matrix": np.zeros(
            (
                n_total_firms,
                n_total_categories,
                n_total_firms,
                n_total_categories,
                n_datatypes,
            )
        ),
        "data_value": np.zeros(
            (n_consumers, n_total_categories, n_total_firms, n_datatypes)
        ),
        "data_held": np.zeros(
            (n_ticks, n_consumers, n_total_categories, n_total_firms, n_datatypes)
        ),
        "privacy_mask": np.ones((n_consumers, n_total_firms)),
    }
