import numpy as np

from .tracking import SimTracker
from .setup_sim import setup_simulation
import model.innovation as inno
from .utility import utility_for_consumers, choose_firms
from .utils import multinomial, min_max_scaler
import model.data_handling as data
from .privacy_scenario import delete_data


def run(
    general_dict={},
    seed_dict={},
    util_weight_dict={},
    innovation_dict={},
    capital_dict={},
    needs_dict={},
    data_dict={},
    category_dict={},
    usage_dict={},
    port_dict={},
    privacy_dict={},
    scenario_dict={},
    openness_dict={},
):

    # unpacking some general parameters
    n_ticks = general_dict["n_ticks"]
    n_consumers = general_dict["n_consumers"]
    n_init_firms = general_dict["n_init_firms"]
    w_logit = util_weight_dict["w_logit"]
    alpha_usage_decay = usage_dict["alpha_usage_decay"]
    data_worth_exp = data_dict["data_worth_exp"]
    qual_diff_param = innovation_dict["qual_diff_param"]
    inno_low = innovation_dict["success_invest_low"]
    inno_high = innovation_dict["success_invest_high"]
    # setting up random number generator
    rng = np.random.RandomState(seed=seed_dict["overall_seed"])

    # setting the scene
    setup_dict = setup_simulation(
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
    )
    capital = setup_dict["capital"]
    ticks_no_capital = setup_dict["ticks_no_capital"]
    usage = setup_dict["usage"]
    ticks_no_usage = setup_dict["ticks_no_usage"]
    quality = setup_dict["quality"]
    firm_privacy_score = setup_dict["firm_privacy_score"]
    consumer_privacy_concern = setup_dict["consumer_privacy_concern"]
    consumer_wealth = setup_dict["consumer_wealth"]
    F_alive = setup_dict["F_alive"]
    need_matrix = setup_dict["need_matrix"]
    category_datatype = setup_dict["category_datatype"]
    n_datatypes = category_datatype.shape[1]
    firm_investment_profile = setup_dict["firm_investment_profile"]
    usage_counter = setup_dict["usage_counter"]
    usage_counter_raw = np.zeros_like(usage_counter)
    data_combination_skill = setup_dict["data_combination_skill"]
    requestable = setup_dict["requestable"]
    requestable_now = np.zeros_like(requestable, dtype=np.int8)
    portability_matrix = setup_dict["portability_matrix"]
    data_value = setup_dict["data_value"]
    data_held = setup_dict["data_held"]
    uninterrupted_usage = np.zeros_like(usage_counter)
    privacy_mask = setup_dict["privacy_mask"]
    inno_new_prod_alpha = innovation_dict["new_product_scaler_alpha"]

    new_firm_new_category_prob = innovation_dict["new_firm_new_category_prob"]

    i_alive = n_init_firms  # highest index of an active firm, plus one
    cat_ever_alive = (quality.sum(axis=0) > 0).astype(int)
    n_total_firms = capital.shape[0]
    n_total_categories = category_dict["n_total_categories"]

    category_total_usage = np.zeros(n_total_categories)
    category_ticks_alive = (quality.sum(axis=0) > 0).astype(int)

    # setting up the tracker
    tracker = SimTracker(
        n_ticks,
        n_total_firms,
        n_total_categories,
        n_consumers,
        quality,
        capital_dict["small"],
    )
    tracker.add_needs(need_matrix)

    scen_tick = -1
    if scenario_dict:
        scen_tick = scenario_dict["scen_tick"]

    for tick in np.arange(0, n_ticks):

        # SCENARIO
        if tick == scen_tick:
            # set up a random number generator for the scenario
            scen_rng = np.random.RandomState(seed_dict["scenario_seed"])
            # choose which firms to shock: the n biggest in terms of capital
            firm_list = np.argsort(capital)[-scenario_dict["scen_number_of_firms"]:]
            # adjust the privacy score of the affected firms
            firm_privacy_score[firm_list] = np.maximum(
                firm_privacy_score[firm_list] - scenario_dict["firm_hit"], 0.05
            )
            # Adjust consumer privacy concern
            consumer_privacy_concern += scen_rng.normal(
                scenario_dict["consumer_hit_mean"],
                scenario_dict["consumer_hit_var"],
                size=n_consumers,
            )
            # some people request a deletion of data and will never use the firm again
            # (consumer, firm)
            data_deleters = np.zeros((n_consumers, n_total_firms))
            data_deleters[:, firm_list] = (
                scen_rng.uniform(size=(n_consumers, len(firm_list)))
                < consumer_privacy_concern[:, None]
            ).astype(int)
            # recalculate the amount of data held by the firms, and its value
            data_held, data_value = delete_data(
                data_deleters, data_held, data_value, tick, data_worth_exp
            )
            # The consumers who have requested data to be deleted by the impacted firms,
            # will never use these firms again
            privacy_mask *= 1 - data_deleters

        if tick > 0:
            # BIRTH OF NEW FIRMS
            # either in new market or in existing market
            num_new_firms = rng.poisson(general_dict["birth_lambda"])
            # make sure we're not running out of firms
            num_new_firms_ = np.minimum(num_new_firms, n_total_firms - i_alive)
            if num_new_firms_ > 0:
                # change aliveness indicator
                F_alive[i_alive : (i_alive + num_new_firms_)] = 1
                # give capital
                capital[i_alive : (i_alive + num_new_firms_)] = capital_dict["small"]
                # entering existing category or make a new one?
                # last_cat = np.max(np.where(cat_ever_alive > 0)[0])
                # remaining_categories = n_total_categories - last_cat - 1
                remaining_categories = (cat_ever_alive == 0).astype(int).sum()
                new_category_count = np.minimum(
                    np.sum(
                        rng.uniform(size=num_new_firms_) < new_firm_new_category_prob
                    ),
                    remaining_categories,
                )
                # assign quality 1 for companies entering a non-existing category
                quality[
                    i_alive + np.arange(new_category_count),
                    np.where(cat_ever_alive == 0)[0][:new_category_count],
                ] = 1
                # deal with companies entering an existing category
                existing_category_count = num_new_firms_ - new_category_count
                if existing_category_count > 0:
                    quality[
                        i_alive
                        + new_category_count
                        + np.arange(existing_category_count)
                    ] += inno.enter_new_category(
                        quality,
                        category_datatype,
                        None,
                        category_total_usage,
                        category_ticks_alive,
                        innovation_dict,
                        qual_diff_param,
                        rng,
                        n=existing_category_count,
                    )
                # more bookkeeping
                ticks_no_usage[i_alive : (i_alive + num_new_firms_)] = 0
                ticks_no_capital[i_alive : (i_alive + num_new_firms_)] = 0
                i_alive += num_new_firms_
                cat_ever_alive = ((cat_ever_alive + quality.sum(axis=0)) > 0).astype(
                    int
                )

            # DEATH OF FIRMS
            # check whether firms should die based on usage/capital
            death_mask = (
                (ticks_no_usage > general_dict["no_usage_ticks_before_death"])
                | (ticks_no_capital > general_dict["no_money_ticks_before_death"])
            ) & F_alive.astype(bool)
            # for tracking
            num_dead_firms = death_mask.astype(int).sum()
            # update alive mask
            F_alive[death_mask] = 0
            # take products off market
            quality[death_mask] = 0
            # no porting from/between dead firms
            portability_matrix[death_mask] = 0
            portability_matrix[:, :, death_mask] = 0
            # also no more requests from/to dead firms
            requestable[death_mask] = 0
            requestable[:, :, death_mask] = 0

            # REQUESTING DATA RIGHTS
            A = (quality > 0).astype(int)
            # what is requestable now?
            A_where_0, A_where_1 = np.where(A)
            requestable_now = data.numba_calc_avail_now(
                requestable, A_where_0, A_where_1
            )
            # first pick datatype to request:
            # (Firm requesting, Datatype) mask for what follows
            firm_dt_avail_for_request = (
                requestable_now.sum(axis=(1, 2, 3)) > 0
            ).astype(int)
            # (firm, dt): how much of each datatype does the firm use
            firm_datatype = (
                (quality > 0).astype(int)[:, :, None] * category_datatype[None, :, :]
            ).sum(axis=1)
            firm_datatype *= firm_dt_avail_for_request
            # (firm, datatype) datatype choice to request
            firm_dt_prob = np.nan_to_num(
                firm_datatype / firm_datatype.sum(axis=-1, keepdims=True)
            )
            firm_dt_choice = multinomial(firm_dt_prob, rng)
            # line below: in case there is no datatype available for requesting, just choose the first ones
            # will be dealt with later
            firm_dt_choice[firm_dt_choice.sum(axis=-1) == 0, 0] = 1
            # choose firm from which data will be requested
            r_dt, c_dt = np.where(firm_dt_choice)
            firm_firm_avail = (
                requestable_now[r_dt, :, :, :, c_dt].sum(axis=(1, 3)) > 0
            ).astype(int)
            firm_firm_prob = np.nan_to_num(
                firm_firm_avail / firm_firm_avail.sum(axis=-1, keepdims=True)
            )
            firm_firm_choice = multinomial(firm_firm_prob, rng)
            # again a hack for firms that can't make any requests
            firm_firm_choice[firm_firm_choice.sum(axis=-1) == 0, 0] = 1
            # Choose which category to import data from
            r_f, c_f = np.where(firm_firm_choice)
            firm_cat_from_avail = (
                requestable_now[r_dt, :, c_f, :, c_dt].sum(axis=1) > 0
            ).astype(int)
            firm_cat_from_prob = np.nan_to_num(
                firm_cat_from_avail / firm_cat_from_avail.sum(axis=-1, keepdims=True)
            )
            firm_cat_from_choice = multinomial(firm_cat_from_prob, rng)
            # same hack again...
            firm_cat_from_choice[firm_cat_from_choice.sum(axis=-1) == 0, 0] = 1
            # Choose which category the import will be made to
            r_cf, c_cf = np.where(firm_cat_from_choice)
            firm_cat_to_avail = requestable_now[r_dt, :, c_f, c_cf, c_dt]
            firm_cat_to_prob = np.nan_to_num(
                firm_cat_to_avail / firm_cat_to_avail.sum(axis=-1, keepdims=True)
            )
            firm_cat_to_choice = multinomial(firm_cat_to_prob, rng)
            # ... and the same hack again
            firm_cat_to_choice[firm_cat_to_choice.sum(axis=-1) == 0, 0] = 1
            r_ct, c_ct = np.where(firm_cat_to_choice)
            # deal with the hacks done above
            request_mask = data.numba_mask_impossible_requests(
                r_ct, c_ct, c_f, c_cf, c_dt, requestable_now
            )
            r_ct, c_ct, c_f, c_cf, c_dt = [
                x[request_mask == 1] for x in [r_ct, c_ct, c_f, c_cf, c_dt]
            ]
            # r_ct: firm requesting the rights to datatype
            # c_ct: category they want to import data to
            # c_f: firm receiving the data request
            # c_cf: category data will be imported from if request is granted
            # c_dt: datatype asked for

            # GRANTING/DENYING DATA RIGHTS
            granting_probs = data.calculate_granting_probs(
                r_ct,
                c_f,
                A,
                openness_dict["openness_lower"],
                openness_dict["openness_upper"],
            )
            granted_mask = rng.uniform(size=granting_probs.shape) < granting_probs
            r_ct_g, c_ct_g, c_f_g, c_cf_g, c_dt_g = [
                x[granted_mask] for x in [r_ct, c_ct, c_f, c_cf, c_dt]
            ]
            # update requestable(also set to zero if request was granted, so we won't ask again)
            requestable = data.numba_update_requestable(
                requestable, r_ct, c_ct, c_f, c_cf, c_dt
            )
            # update the portability matrix
            portability_matrix = data.numba_update_portability_matrix(
                portability_matrix, r_ct_g, c_ct_g, c_f_g, c_cf_g, c_dt_g
            )

            # INNOVATION IN EXISTING FIRMS
            # money to be invested - zero for firms that do no yet exist
            capital_to_invest = np.maximum(
                np.minimum(capital, innovation_dict["invest_cap"]), 0
            )
            capital = capital - capital_to_invest
            # Either invest in a existing product; in an existing category which they don't have a product in;
            # or in an non-existing category
            firm_investment_profile_ = firm_investment_profile
            # if a firm already has all categories, middle option can't be chosen
            firm_investment_profile_[
                (quality[:, cat_ever_alive] == 0).sum(axis=-1) == 0, 1
            ] = 0
            firm_investment_profile_[
                (quality[:, cat_ever_alive] == 0).sum(axis=-1) == 0, 2
            ] = (
                1
                - firm_investment_profile_[
                    (quality[:, cat_ever_alive] == 0).sum(axis=-1) == 0, 0
                ]
            )
            # no more new categories to expand into
            firm_investment_profile_[:, -1] *= 1 - (
                cat_ever_alive.sum() == n_total_categories
            ).astype(int)
            firm_investment_profile_ = (
                firm_investment_profile_
                / firm_investment_profile_.sum(axis=-1, keepdims=True)
            )
            investment_choice = (
                multinomial(firm_investment_profile_, rng) * F_alive[:, None]
            )

            # Existing product that investment will be in
            invest_prob = inno.invest_utility_existing(
                quality, category_total_usage, tick, innovation_dict
            )
            invest_product = (
                multinomial(invest_prob, rng) * investment_choice[:, 0][:, None]
            )
            # calculating the data investment
            # (firm, datatypes)
            rel_datatypes = invest_product.dot(category_datatype)
            invest_data_value_base = (rel_datatypes[None, None, :, :] * data_value).sum(
                axis=(0, 1)
            )
            invest_data_value = inno.apply_data_skill(
                invest_data_value_base, data_combination_skill
            )
            # investment = min_max_scaler(capital_to_invest * invest_data_value) * investment_choice[:, 0]
            investment = inno.investment_scaler(
                capital_to_invest * invest_data_value, 0, 1, inno_new_prod_alpha
            )  # this is number between 0 and 1
            # calculating the gain in quality
            extra_quality = inno.product_quality_update(
                quality, investment, innovation_dict["alpha_f"]
            )
            quality += extra_quality * invest_product

            # Firms going into a category they haven't developed before
            # get the potential added quality - IF firms succeed
            potential_added_quality = inno.enter_new_category(
                quality,
                category_datatype,
                investment_choice[:, 1:],
                category_total_usage,
                category_ticks_alive,
                innovation_dict,
                qual_diff_param,
                rng,
            )
            assert (potential_added_quality * quality == 0).all(), (
                np.where(potential_added_quality * quality),
                investment_choice[1],
                firm_investment_profile_[1],
                quality[1],
                cat_ever_alive,
                n_total_categories,
            )
            # get the data investment for the product under development
            rel_datatypes = (potential_added_quality > 0).dot(category_datatype)
            invest_data_value_base = (rel_datatypes[None, None, :, :] * data_value).sum(
                axis=(0, 1)
            )
            invest_data_value = inno.apply_data_skill(
                invest_data_value_base, data_combination_skill
            )
            investment = inno.investment_scaler(
                capital_to_invest * invest_data_value,
                inno_low,
                inno_high,
                inno_new_prod_alpha,
            )  # this is a probability
            success = (
                (rng.uniform(size=(n_total_firms)) < investment).astype(int)
                * investment_choice[:, 1:].sum(axis=-1)
                * F_alive
            )
            quality += potential_added_quality * success[:, None]
            cat_ever_alive = ((cat_ever_alive + quality.sum(axis=0)) > 0).astype(int)

        # CONSUMERS USING A PRODUCT
        # decide which product categories consumers will use in this tick (consumers, categories
        #  - provided the category exists
        usage_product_mask = (need_matrix > rng.uniform(size=need_matrix.shape)).astype(
            int
        ) * (quality.sum(axis=0) > 0).astype(int)[None, :]
        # utility (consumer, category, firm)
        utility_ = utility_for_consumers(
            quality,
            usage,
            usage_counter,
            consumer_privacy_concern,
            firm_privacy_score,
            util_weight_dict,
        )
        # choosing a firm (consumer, category, firm)
        usage_firm = choose_firms(utility_, quality, w_logit, privacy_mask, rng)
        # mask for products used (consumer, category, firm)
        usage = usage_firm * usage_product_mask[:, :, None].astype(int)

        # BOOKKEEPING: CUSTOMER DATA + PAYMENT
        # update capital: agents uniformly distribute money over products consumers in tick
        prod_per_consumer = np.maximum(usage.sum(axis=(1, 2)), 1)
        capital += (
            np.nan_to_num(consumer_wealth / prod_per_consumer)[:, None]
            * usage.sum(axis=1)
        ).sum(axis=0)
        data_value *= np.exp(-data_worth_exp)
        usage_cons, usage_cat, usage_firm = np.where(usage)
        data_held, data_value = data.update_data_stuff(
            data_held,
            data_value,
            usage_cons,
            usage_cat,
            usage_firm,
            category_datatype,
            tick,
            n_datatypes,
        )
        has_used_mask = (usage.sum(axis=2, keepdims=True) > 0) * np.ones(
            n_total_firms
        ).astype(bool)[None, None, :]
        uninterrupted_usage[has_used_mask] *= usage[has_used_mask]
        uninterrupted_usage[has_used_mask] += usage[has_used_mask]
        usage_counter_raw += usage
        usage_counter *= np.exp(-alpha_usage_decay)
        usage_counter += usage
        category_total_usage += usage.sum(axis=(0, 2))
        category_ticks_alive[quality.sum(axis=0) > 0] += 1

        # PORTING
        # Decision to port data: at nth consecutive usage, port everything that's portable
        cons_, cat_, firm_ = np.where(uninterrupted_usage == port_dict["n_port"])
        if len(cons_) > 0:
            PM = np.swapaxes(portability_matrix[firm_, cat_], 1, 2) * (
                data_value[cons_] > 0
            ).astype(int)
            if PM.sum() > 0:
                data_held, data_value = data.port(
                    cons_, cat_, firm_, data_held, data_value, tick, PM
                )

        # update no usage/no capital trackers
        ticks_no_usage[usage.sum(axis=(0, 1)) == 0] += 1
        ticks_no_usage[usage.sum(axis=(0, 1)) > 0] = 0
        ticks_no_capital[capital < capital_dict["capital_cutoff"]] += 1
        ticks_no_capital[capital >= capital_dict["capital_cutoff"]] = 0

        tracker.update(
            tick,
            (
                quality,
                capital,
                usage,
                0 if tick == 0 else num_new_firms_,
                0 if tick == 0 else num_dead_firms,
                F_alive,
                0 if tick == 0 else investment_choice,
                0 if tick == 0 else success,
                consumer_privacy_concern,
                firm_privacy_score,
                0 if tick == 0 else r_ct_g,
                0 if tick == 0 else capital_to_invest * invest_data_value,
            ),
        )
    output = tracker.gather_output()
    return output
