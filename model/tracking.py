import numpy as np
import pandas as pd
from scipy.stats import rankdata


class SimTracker(object):
    def __init__(
        self, n_ticks, n_firms, n_categories, n_consumers, quality, small_start_capital
    ):
        self._quality = np.zeros((n_ticks, n_firms, n_categories))
        self._capital = np.zeros((n_ticks, n_firms))
        self._usage = np.zeros((n_ticks, n_firms, n_categories))
        self._usage_consumer = np.zeros((n_ticks, n_consumers, n_firms))
        self._new_firms = np.zeros((n_ticks))
        self._dead_firms = np.zeros((n_ticks))
        self._live_firms = np.zeros((n_ticks, n_firms))
        self._cat_new_firms = (quality > 0).sum(axis=0)
        self._cat_dead_firms = np.zeros((n_categories), dtype=int)
        self._new_products_existing_cat = np.zeros((n_ticks), dtype=int)
        self._new_products_new_cat = np.zeros((n_ticks), dtype=int)
        self._investment_choices = np.zeros((n_ticks, n_firms, 3))
        self._investment_success = np.zeros((n_ticks, n_firms))
        self._concern = np.zeros((n_ticks, n_consumers))
        self._privacy_score = np.zeros((n_ticks, n_firms))
        self._start_tick_new_firms = -1 * np.ones((n_firms), dtype=int)
        self._start_tick_new_firms[quality.sum(axis=-1) > 0] = 0
        self._first_year_usage = np.zeros((n_firms))
        self._first_year_requests_granted = np.zeros((n_firms))
        self._small_start_capital = small_start_capital
        self._success_prob = np.zeros((n_ticks, n_firms, 2))

    def update(self, tick, data):
        F_qual_tick, F_cap_tick, F_usage_tick, num_new_firms, num_dead_firms, F_alive, \
            investment, success, concern, ps, r_ct_g, success_prob = data

        self._usage[tick] = F_usage_tick.sum(axis=0).T
        self._usage_consumer[tick] = (F_usage_tick.sum(axis=1) > 0).astype(int)
        self._quality[tick] = F_qual_tick
        self._capital[tick] = F_cap_tick
        self._live_firms[tick] = F_alive
        self._concern[tick] = concern
        self._privacy_score[tick] = ps

        if tick > 0:
            self._new_firms[tick] = num_new_firms
            self._dead_firms[tick] = num_dead_firms
            self._investment_choices[tick] = investment
            self._investment_success[tick] = success
            self._success_prob[tick] = investment[:, 1:] * success_prob[:, None]
            assert (
                F_alive.sum()
                == self._live_firms[tick - 1].sum() + num_new_firms - num_dead_firms
            ), (
                tick,
                F_alive.sum(),
                self._live_firms[tick - 1].sum(),
                num_new_firms,
                num_dead_firms,
            )
        if tick > 0:
            # counting firms entering/leaving categories
            diff_qual = (F_qual_tick > 0).astype(int) - (
                self._quality[tick - 1] > 0
            ).astype(int)
            self._cat_new_firms += (diff_qual == 1).astype(int).sum(axis=0)
            self._cat_dead_firms += (diff_qual == -1).astype(int).sum(axis=0)
            self._start_tick_new_firms[
                ((diff_qual == 1).astype(int).sum(axis=-1) > 0)
                & (self._start_tick_new_firms == -1)
            ] = tick
            self._first_year_usage += (
                (self._start_tick_new_firms > 0)
                & (tick - self._start_tick_new_firms < 12)
            ).astype(int) * self._usage[tick].sum(axis=-1)
            self._first_year_requests_granted += (
                (self._start_tick_new_firms > 0)
                & (tick - self._start_tick_new_firms < 12)
            ).astype(int) * np.bincount(
                r_ct_g, minlength=self._first_year_requests_granted.shape[0]
            )

            # counting total entry into new and existing categories
            new_products = ((F_qual_tick > 0) & (self._quality[tick - 1] == 0)).astype(
                int
            )
            existing_cats = (self._quality[tick - 1].sum(axis=0) > 0).astype(int)
            new_cats = (
                (F_qual_tick.sum(axis=0) > 0)
                & (self._quality[tick - 1].sum(axis=0) == 0)
            ).astype(int)
            self._new_products_new_cat[tick] = (new_products * new_cats[None, :]).sum()
            self._new_products_existing_cat[tick] = (
                new_products * existing_cats[None, :]
            ).sum()

    def add_needs(self, need_matrix):
        self._need_matrix = need_matrix

    def gather_output(self):
        n_ticks, n_firms, n_categories = self._quality.shape

        # data for firm specialisation plot - only on firms that have ever existed
        active = self._quality[:, self._quality.sum(axis=(0, 2)) > 0, :]
        active = (active[-12:].sum(axis=0) > 0).astype(int)
        bins = np.bincount(active.sum(axis=-1))
        total = bins.sum()
        self._firm_specialisation_df = pd.DataFrame(
            {"bins": np.arange(len(bins)), "perc": np.round(bins / total * 100, 1)}
        )

        # data for market concentration
        # (firm, category)
        cons_count_mat = self._usage[-12:].sum(axis=0)
        # (category)
        total_per_cat = cons_count_mat.sum(axis=0)
        # (category)
        top3_cons_count = (np.sort(cons_count_mat, axis=0)[::-1][:3]).sum(axis=0)
        perc = np.round(100 * top3_cons_count / total_per_cat, 1)
        res_df = pd.DataFrame({"category": np.arange(n_categories), "consumer": perc})
        # how many firms still active
        still_active_last_ticks = ((self._quality[-12:] > 0).sum(axis=0) > 0).sum(
            axis=0
        )
        sa_df = pd.DataFrame(
            {
                "category": np.arange(n_categories),
                "firms_active": still_active_last_ticks,
            }
        )
        res_df = pd.merge(res_df, sa_df, on="category")
        self._market_share_df = res_df

        # data for market concentration over time
        # (tick, firm, category)
        cons_count_mat = self._usage
        # (tick, category)
        total_per_cat = cons_count_mat.sum(axis=1)
        # (tick, category)
        top3_cons_count = np.sort(cons_count_mat, axis=1)[:, ::-1][:, :3].sum(axis=1)
        perc = np.round(100 * top3_cons_count / total_per_cat, 1)
        res_df = (
            pd.DataFrame(
                perc, index=np.arange(n_ticks), columns=np.arange(n_categories)
            )
            .stack()
            .reset_index()
            .rename(columns={"level_0": "tick", "level_1": "category", 0: "consumer"})
        )
        # how many firms are active in each tick? (tick, category)
        active_count = (self._quality > 0).astype(int).sum(axis=1)
        sa_df = (
            pd.DataFrame(
                active_count, index=np.arange(n_ticks), columns=np.arange(n_categories)
            )
            .stack()
            .reset_index()
            .rename(
                columns={"level_0": "tick", "level_1": "category", 0: "firms_active"}
            )
        )
        res_df = pd.merge(res_df, sa_df, on=["category", "tick"])
        self._market_share_timeline_df = res_df

        # data for complimentarity
        firm_count = np.bincount(
            (self._usage_consumer[-12:].sum(axis=0) > 0).astype(int).sum(axis=-1)
        )
        self._complimentarity_df = pd.DataFrame(
            {
                "bins": np.arange(len(firm_count)),
                "perc": np.round(firm_count / firm_count.sum() * 100, 1),
            }
        )

        # data for consumer welfare
        max_qual_category = np.max(self._quality, axis=(0, 1))
        median_needs = np.median(self._need_matrix, axis=0)
        cat_ranking = rankdata(median_needs)
        mean_num_firms = (self._quality[-12:] > 0).astype(int).sum(axis=(0, 1)) / 12
        # (category) overall usage over time and firms
        usage_count = self._usage.sum(axis=(0, 1))
        # (category) how many ticks was category available?
        tick_count = (self._quality.sum(axis=1) > 0).astype(int).sum(axis=0)
        mean_usage_per_tick = usage_count / tick_count
        self._welfare_df = pd.DataFrame(
            {  #'rank_': cat_ranking, 'median_': np.round(median_needs, 4),
                "quality": np.round(max_qual_category, 2),
                "num_firms": np.round(mean_num_firms, 1),
                "category": np.arange(n_categories),
                "mean_usage_per_tick": np.round(mean_usage_per_tick, 0),
            }
        )
        # data for consumer welfare all companies
        self._welfare_df_all_products = pd.DataFrame(
            {
                "median_": (
                    median_needs[None, :] * np.ones((n_firms, n_categories))
                ).flatten(),
                "quality": np.max(self._quality[-12:], axis=0).flatten(),
                "num_firms": (
                    mean_num_firms[None, :] * np.ones((n_firms, n_categories))
                ).flatten(),
                "category": (
                    np.arange(n_categories)[None, :] * np.ones((n_firms, n_categories))
                ).flatten(),
            }
        )
        self._welfare_df_all_products = self._welfare_df_all_products.loc[
            self._welfare_df_all_products.quality > 0
        ]

        # gathering the investment choices
        self._invest_df = pd.DataFrame(
            self._investment_choices.sum(axis=1),
            index=range(n_ticks),
            columns=["Existing prod", "New prod", "New cat"],
        )

        # gathering the innovation success
        self._new_prod_success = pd.DataFrame(
            (self._investment_choices * self._investment_success[:, :, None]).sum(
                axis=1
            ),
            index=range(n_ticks),
            columns=["Existing prod", "New prod", "New cat"],
        )

        # gather overall usage of all products a firm offers
        self._firm_usage_df = pd.DataFrame(
            self._usage.sum(axis=-1), columns=np.arange(n_firms), index=range(n_ticks)
        )

        # gather the evolution of the privacy score of firms
        self._ps_score_evo = pd.DataFrame(
            self._privacy_score, columns=np.arange(n_firms), index=range(n_ticks)
        )

        # gathering the evolution of privacy concern over time
        n_consumers = self._concern.shape[1]
        self._concern_evo = pd.DataFrame(
            self._concern, columns=np.arange(n_consumers), index=range(n_ticks)
        )

        # evolution of quality over time
        mi = pd.MultiIndex.from_product(
            [np.arange(n_ticks), np.arange(n_firms), np.arange(n_categories)],
            names=["tick", "firm", "category"],
        )
        quality_series = pd.Series(index=mi, data=self._quality.flatten())
        quality_series = quality_series[quality_series > 0]
        self._quality_df = (
            quality_series.to_frame().reset_index().rename(columns={0: "quality"})
        )

        # growth of firms during the first year (only for firms > 12mo)
        growth_mask_firms = (self._start_tick_new_firms > 0) & (
            self._start_tick_new_firms + 12 < n_ticks
        )
        self._year_growth_df = pd.DataFrame(
            self._capital[
                self._start_tick_new_firms[growth_mask_firms] + 12,
                np.arange(n_firms, dtype=int)[growth_mask_firms],
            ]
            - self._small_start_capital,
            index=np.arange(n_firms)[growth_mask_firms],
            columns=["Growth"],
        )
        self._year_growth_df_2 = pd.DataFrame(
            self._first_year_usage[growth_mask_firms],
            index=np.arange(n_firms, dtype=int)[growth_mask_firms],
            columns=["Growth"],
        )

        # data for data request plot
        self._data_plot_df = pd.DataFrame(
            {
                "requests": self._first_year_requests_granted[growth_mask_firms],
                "num_cat": self._quality[
                    (self._start_tick_new_firms + 12)[growth_mask_firms],
                    np.where(growth_mask_firms)[0],
                ].sum(axis=-1),
            }
        )

        # succesful innovations
        # type 0 = new product in existing cat; 1 = new product in new cat
        tick_, firm_, type_ = np.where(self._success_prob > 0)
        self._success_prob_df = pd.DataFrame(
            {
                "tick": tick_,
                "firm": firm_,
                "type": type_,
                "prob": self._success_prob[tick_, firm_, type_],
            }
        )

        return {
            'quality': self._quality_df,
            'capital': pd.DataFrame(self._capital, index=np.arange(n_ticks), columns=np.arange(n_firms)),
            'usage': self._usage,
            'new_firms': self._new_firms,
            'dead_firms': self._dead_firms,
            'ps_score_evo': self._ps_score_evo,
            'concern_evo': self._concern_evo,
            'firm_usage_df': self._firm_usage_df,
            'new_prod_success': self._new_prod_success,
            'invest_df': self._invest_df,
            'success_prob_df': self._success_prob_df,
            'welfare_df_all_products': self._welfare_df_all_products,
            'cat_new_firms': self._cat_new_firms,
            'cat_dead_firms': self._cat_dead_firms,
            'year_growth_df': self._year_growth_df_2,
            'new_products_existing_cat': pd.Series(self._new_products_existing_cat),
            'new_products_new_cat': pd.Series(self._new_products_new_cat),
            'market_share_timeline_df': self._market_share_timeline_df,
            "welfare_df": self._welfare_df,
            "innovation_df": pd.concat(
                [
                    pd.DataFrame(
                        self._new_products_new_cat,
                        index=np.arange(self._new_products_new_cat.shape[0]),
                        columns=["new"],
                    ),
                    pd.DataFrame(
                        self._new_products_existing_cat,
                        index=np.arange(self._new_products_existing_cat.shape[0]),
                        columns=["existing"],
                    ),
                ],
                axis=1,
            ),
            "complimentarity_df": self._complimentarity_df,
            "firm_specialisation_df": self._firm_specialisation_df,
            "market_share_df": self._market_share_df,
            "cat_entry_and_exit_df": pd.DataFrame(
                {"entry": self._cat_new_firms, "exit": self._cat_dead_firms}
            ),
            "data_request_plot_df": self._data_plot_df,
        }
