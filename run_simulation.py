from model.simulation import run
from model.figures import *

import plotly.offline as py
import os
import click
import yaml


def make_directory_if_doesnt_exist(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def read_yaml(filename):
    with open(filename, "r") as stream:
        return yaml.load(stream)


def save_plot(plot, directory, filename):
    py.plot(plot, filename=os.path.join(directory, filename))


@click.command()
@click.option(
    "--input_yaml",
    "-i",
    help="Path to yaml with parameters",
    default="model_parameters.yaml",
)
@click.option("--output_dir", "-o", help="Output directory", default="./tests")
def create_outputs(input_yaml, output_dir):
    make_directory_if_doesnt_exist(output_dir)
    yam = read_yaml(input_yaml)
    out = run(**yam)

    quality = out["quality"]
    save_plot(plot_quality_by_category(quality), output_dir, "quality_by_category.html")
    save_plot(plot_quality_by_firm(quality), output_dir, "quality_by_firm.html")

    cat_entry_and_exit_df = out["cat_entry_and_exit_df"]
    save_plot(plot_market_entry(cat_entry_and_exit_df), output_dir, "market_entry.html")

    market_share_df = out["market_share_df"]
    save_plot(
        plot_market_concentration(market_share_df),
        output_dir,
        "market_concentration.html",
    )

    firm_specialisation_df = out["firm_specialisation_df"]
    save_plot(
        plot_firm_specialisation(firm_specialisation_df),
        output_dir,
        "firm_specialisation.html",
    )

    complimentarity_df = out["complimentarity_df"]
    save_plot(
        plot_complimentarity(complimentarity_df), output_dir, "complimentarity.html"
    )

    new_products_new_cat = out["new_products_new_cat"]
    new_products_existing_cat = out["new_products_existing_cat"]
    save_plot(
        plot_new_products(new_products_new_cat, new=True),
        output_dir,
        "new_products.html",
    )
    save_plot(
        plot_new_products(new_products_existing_cat),
        output_dir,
        "existing_products.html",
    )

    welfare_df = out["welfare_df"]
    save_plot(plot_welfare(welfare_df), output_dir, "welfare.html")

    market_share_timeline_df = out["market_share_timeline_df"]
    save_plot(
        plot_market_share_timeline(market_share_timeline_df),
        output_dir,
        "market_share.html",
    )

    firm_usage_df = out["firm_usage_df"]
    save_plot(plot_firm_engagement(firm_usage_df), output_dir, "firm_usage.html")

    concern_evo = out["concern_evo"]
    save_plot(plot_concern(concern_evo), output_dir, "concern_evo.html")

    privacy_score_evo = out["ps_score_evo"]
    save_plot(plot_privacy_score(privacy_score_evo), output_dir, "privacy_score.html")

    capital = out["capital"]
    save_plot(plot_capital(capital), output_dir, "capital.html")

    data_request_plot_df = out["data_request_plot_df"]
    save_plot(
        plot_data_requests(data_request_plot_df), output_dir, "data_requests.html"
    )


if __name__ == "__main__":
    create_outputs()
