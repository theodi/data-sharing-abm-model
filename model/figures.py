import plotly.graph_objs as go
import numpy as np

from .beta_distr import get_beta_params
from scipy.stats import beta


import matplotlib.pyplot as plt

# get some nice colours
tab10 = plt.get_cmap("tab10").colors
tab10 = list(tuple(x * 255 for x in tab10[i]) for i in range(len(tab10)))
tab10 = ["rgb" + str(x) for x in tab10]


def plot_beta(m, var):
    """
    plot a single beta function with mode m and variance var
    """
    a, b = get_beta_params(m, var)
    xs = np.linspace(0, 1, 1000)
    return go.Figure(data=[go.Scatter(x=xs, y=beta.pdf(xs, a, b), mode="line")])


def plot_quality_by_category(quality_df):
    """
    evolution of the quality of all products over time, grouped by category
    """
    data = []
    for i, cat in enumerate(quality_df.category.unique()):
        df_ = quality_df.loc[quality_df.category == cat].sort_values(
            "tick", ascending=True
        )
        for j, firm in enumerate(df_.firm.unique()):
            df_1 = df_.loc[df_.firm == firm]
            data += [
                go.Scatter(
                    x=df_1.tick,
                    y=df_1.quality,
                    mode="lines",
                    line={"color": tab10[np.mod(i, 10)]},
                    showlegend=True if j == 0 else False,
                    hovertext="category {} firm {}".format(cat, firm),
                    legendgroup=str(cat),
                    name="category {}".format(cat),
                )
            ]
    layout = go.Layout(title="Evolution of quality (coloured by category)")
    return go.Figure(data=data, layout=layout)


def plot_capital(capital_df):
    """
    evolution of the capital of firms over time
    """
    data = [
        go.Scatter(x=capital_df.index, y=capital_df[c], name="firm " + str(c))
        for c in capital_df.columns
    ]
    layout = go.Layout(title="Capital over time")
    return go.Figure(data=data, layout=layout)


def plot_quality_by_firm(quality_df):
    """
    evolution of the quality of all products over time, grouped by firm
    """
    data = []
    for i, firm in enumerate(quality_df.firm.unique()):
        df_ = quality_df.loc[quality_df.firm == firm].sort_values("tick")
        for j, cat in enumerate(df_.category.unique()):
            df_1 = df_.loc[df_.category == cat]
            data += [
                go.Scatter(
                    x=df_1.tick,
                    y=df_1.quality,
                    mode="lines",
                    line={"color": tab10[np.mod(i, 10)]},
                    showlegend=True if j == 0 else False,
                    hovertext="firm {} category {}".format(firm, cat),
                    legendgroup=str(firm),
                    name="firm {}".format(firm),
                )
            ]
    layout = go.Layout(title="Evolution of quality (coloured by firm)")
    return go.Figure(data=data, layout=layout)


def plot_market_entry(cat_entry_and_exit_df):
    """
    A plot with the total number of firms which have entered/left the categories
    during the simulation
    """
    xs = cat_entry_and_exit_df.index
    new_per_cat = cat_entry_and_exit_df.entry.astype(int)
    dead_per_cat = cat_entry_and_exit_df.exit.astype(int)
    data = [
        go.Bar(
            y=xs,
            x=new_per_cat,
            orientation="h",
            showlegend=False,
            hoverinfo="text",
            hovertext=[
                "{} entries in category {}".format(x, y)
                for x, y in zip(new_per_cat, np.arange(len(new_per_cat)))
            ],
            marker={"color": "#FF6700"},
        ),
        go.Bar(
            y=xs,
            x=-dead_per_cat,
            orientation="h",
            showlegend=False,
            hoverinfo="text",
            hovertext=[
                "{} exits in category {}".format(x, y)
                for x, y in zip(dead_per_cat, np.arange(len(new_per_cat)))
            ],
            marker={"color": "#FF6700"},
        ),
        go.Bar(
            y=xs,
            x=new_per_cat - dead_per_cat,
            orientation="h",
            showlegend=False,
            hoverinfo="text",
            hovertext=[
                "{} net entries in category {}".format(x, y)
                for x, y in zip(new_per_cat - dead_per_cat, np.arange(len(new_per_cat)))
            ],
            marker={"color": "#993d00"},
        ),
    ]
    layout = go.Layout(title="Market entry and exit per category", barmode="overlay")
    return go.Figure(data=data, layout=layout)


def plot_market_concentration(res_df):
    """
    plot of the market share of the top three firmsper category during the last year
    """
    data = [
        go.Bar(
            x=res_df.category.values.astype(int),
            y=res_df.consumer,
            hoverinfo="text",
            hovertext=[
                "{}%, total {} firms".format(x, y)
                for x, y in zip(np.round(res_df.consumer, 1), res_df.firms_active)
            ],
            marker={"opacity": [1 if x > 3 else 0.5 for x in res_df.firms_active]},
        )
    ]
    layout = go.Layout(
        title="Market share of top three firms in each category",
        yaxis={"title": "% of category served by top three"},
        xaxis={"title": "Category"},
    )
    return go.Figure(data=data, layout=layout)


def plot_market_share_timeline(df):
    """
    timeline of the percentage of market share owned by the top
    three firms in each category
    """
    data = []
    for cat in df.category.unique():
        df_ = df.loc[df.category == cat]
        data += [
            go.Scatter(
                x=df_.tick,
                y=df_.consumer,
                hoverinfo="text",
                mode="lines",
                name="category {}".format(cat),
                hovertext=[
                    "category {}: {}%, total {} firms".format(cat, x, y)
                    for x, y in zip(np.round(df_.consumer, 1), df_.firms_active)
                ],
            )
        ]
    layout = go.Layout(
        title="Market share of top three firms in each category - timeline",
        yaxis={"title": "% of category served by top three"},
        xaxis={"title": "Category"},
    )
    return go.Figure(data=data, layout=layout)


def plot_firm_specialisation(df):
    """
    histogram of the number of categories firms are active in
    """
    data = [
        go.Bar(
            x=df.bins,
            y=df.perc,
            hoverinfo="text",
            hovertext=["{}%".format(x) for x in np.round(df.perc, 1)],
        )
    ]
    layout = go.Layout(
        title="Firm activity profile",
        xaxis={"title": "Number of categories firms are active in"},
        yaxis={"title": "Frequency"},
    )
    return go.Figure(data=data, layout=layout)


def plot_complimentarity(df):
    """
    plot of the distribution of number of firms consumers
    have used in the last year
    """
    data = [
        go.Bar(
            x=df.bins,
            y=df.perc,
            hoverinfo="text",
            hovertext=["{}%".format(x) for x in np.round(df.perc, 1)],
        )
    ]
    layout = go.Layout(
        title="Consumer usage profile",
        xaxis={"title": "Number of firms used by consumers in the last year"},
        yaxis={"title": "Frequency"},
    )
    return go.Figure(data=data, layout=layout)


def plot_new_products(counter, new=False):
    """
    plot of the number of products being released during the simulation,
    in either new of existent categories
    """
    ticks = np.arange(len(counter)) + 1
    data = [go.Scatter(x=ticks, y=np.cumsum(counter), showlegend=False)]
    layout = go.Layout(
        title="Cumulative number of products being released in {} categories".format(
            "new" if new else "existing"
        ),
        xaxis={"title": "Month in simulation"},
        yaxis={"title": ""},
    )
    return go.Figure(data=data, layout=layout)


def plot_welfare(df):
    """
    plot of the highest quality available in a category, at the end of the run_simulation,
    ordered by the mean usage per tick
    size of dots indicates the number of firms active in that category
    """
    def scaler(vals, min=4, max=10):
        return min + (vals - np.min(vals)) / np.max(vals) * (max - min)

    data = [
        go.Scatter(
            x=df.mean_usage_per_tick,
            y=df.quality,
            mode="markers",
            marker={"size": scaler(df.num_firms.values)},
            hoverinfo="text",
            showlegend=False,
            hovertext=[
                "category {}, offered by {} firms".format(x, y)
                for x, y in zip(df.category.values, np.round(df.num_firms.values, 1))
            ],
        )
    ]
    layout = go.Layout(
        title="(Highest) quality per category", yaxis={"title": "Quality"}
    )
    return go.Figure(data=data, layout=layout)


def plot_investment_choices(invest_df):
    """
    plot of the type of investment choices of firms over time
    """
    data = [
        go.Scatter(x=invest_df.index, y=invest_df[c], name=c) for c in invest_df.columns
    ]
    layout = go.Layout(title="investment choices")
    return go.Figure(data=data, layout=layout)


def plot_investment_success(df):
    """
    plot of investment succes over time
    """
    data = [go.Scatter(x=df.index, y=df[c], name=c) for c in df.columns]
    layout = go.Layout(title="investment success into new product")
    return go.Figure(data=data, layout=layout)


def plot_firm_engagement(df):
    """
    a plot of usage over time
    """
    data = [go.Scatter(x=df.index, y=df[c], name=c) for c in df.columns]
    layout = go.Layout(title="Firm: number of products used over time")
    return go.Figure(data=data, layout=layout)


def plot_concern(df):
    """
    a plot of the consumer privacy concern over time. Each trace is a consumer
    """
    data = [go.Scatter(x=df.index, y=df[c], name=c) for c in df.columns]
    layout = go.Layout(title="Privacy concern over time")
    return go.Figure(data=data, layout=layout)


def plot_privacy_score(df):
    """
    Each trace is the privacy score of a firm over time
    """
    data = [go.Scatter(x=df.index, y=df[c], name=c) for c in df.columns]
    layout = go.Layout(title="Privacy score over time")
    return go.Figure(data=data, layout=layout)


def plot_data_requests(df):
    """
    Scatter plot of the number of categories new firms are active in after one year
    vs the number of data requests that were granted to them in their first year
    """
    n = df.shape[0]
    data = [
        go.Scatter(
            x=df.requests
            + np.random.RandomState(0).uniform(low=-0.05, high=0.05, size=n),
            y=df.num_cat
            + np.random.RandomState(48579438).uniform(low=-0.05, high=0.05, size=n),
            mode="markers",
        )
    ]
    layout = go.Layout(
        title="First year for new firms",
        xaxis={
            "title": "Number of data requests granted to a new firm in its first year"
        },
        yaxis={"title": "Number of categories a new firm is active in after one year"},
    )
    return go.Figure(data=data, layout=layout)
