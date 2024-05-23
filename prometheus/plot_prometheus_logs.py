import csv
import requests
import sys
import numpy as np
import argparse
import pandas as pd
import matplotlib.pyplot as plt


class PrometheusClient:
    def __init__(self, server_url, model_name):
        self.server_url = server_url
        self.model_name = model_name
        self.query_api_url = f"{self.server_url}/api/v1/query"

    def get_metric(self, metric_name, time_window_expr):
        query_str = f'{metric_name}{{model_name="{self.model_name}"}}{time_window_expr}'
        response = requests.get(self.query_api_url, params={"query": query_str})
        return response.json()["data"]["result"]


def plot_hist(
    prom: PrometheusClient,
    metric_str: str,
    time_window_expr: str,
    include_inf=True,
    ax=None,
    normalize=False,
):
    results = prom.get_metric(metric_str, time_window_expr)

    histogram = {}
    for result in results:
        vals_dict = {t: int(v) for t, v in result["values"]}
        histogram[result["metric"]["le"]] = vals_dict

    df = pd.DataFrame.from_dict(histogram)
    df["0.0"] = 0

    if not include_inf:
        df = df.drop(columns=["+Inf"])

    df = df.reindex(
        sorted(df.columns, key=lambda x: int(float(x)) if x != "+Inf" else 1e39), axis=1
    )
    df = df.diff(axis=0).diff(axis=1)
    df = df.drop(columns=["0.0"])
    df.index = (df.index - df.index.min()).astype(int)

    if normalize:
        df = df.div(df.sum(axis=1), axis=0)

    colors = plt.cm.GnBu(np.linspace(0, 1, df.shape[1]))

    print(df)

    df.plot.bar(stacked=True, color=colors, ax=ax)
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize="small")


def plot_iter_tokens_hist(prom: PrometheusClient, time_window_expr: str, **kwargs):
    plot_hist(
        prom,
        "vllm:iteration_tokens_total_bucket",
        time_window_expr,
        include_inf=False,
        **kwargs,
    )


def plot_iter_tpot_hist(prom: PrometheusClient, time_window_expr: str, **kwargs):
    plot_hist(
        prom,
        "vllm:time_per_output_token_seconds_bucket",
        time_window_expr,
        include_inf=True,
        **kwargs,
    )


def plot_iter_ttft_hist(prom: PrometheusClient, time_window_expr: str, **kwargs):
    plot_hist(
        prom, "vllm:time_to_first_token_seconds_bucket", time_window_expr, **kwargs
    )


def plot_iter_tokens_sum_count(prom: PrometheusClient, time_window_expr: str, ax=None):
    total_sum_results = prom.get_metric(
        "vllm:iteration_tokens_total_sum", time_window_expr
    )
    total_count_results = prom.get_metric(
        "vllm:iteration_tokens_total_count", time_window_expr
    )

    assert len(total_sum_results) == 1
    assert len(total_count_results) == 1

    df_dict = {
        "total_tokens": {t: int(v) for t, v in total_sum_results[0]["values"]},
        "total_iters": {t: int(v) for t, v in total_count_results[0]["values"]},
    }

    df = pd.DataFrame.from_dict(df_dict)
    df = df.diff()
    df.index = (df.index - df.index.min()).astype(int)
    print(df)

    plt.plot(df.loc[:, "total_tokens"].values)
    plt.xticks(range(len(df)), df.index.values)  # Restore xticks

    print(total_sum_results[0]["values"])
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize="small")


def hide_every_n_xticks(ax, n):
    xticks = ax.xaxis.get_major_ticks()
    for i in range(len(xticks)):
        if i % n != 0:
            xticks[i].set_visible(False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark the online serving throughput."
    )
    parser.add_argument(
        "time_window_expr",
        type=str,
        default="[1h]",
    )
    parser.add_argument(
        "--prometheus_url",
        type=str,
        help="Prometheus server URL.",
        default="http://localhost:9090",
    )
    args = parser.parse_args()

    prom = PrometheusClient(
        server_url=args.prometheus_url,
        model_name="neuralmagic/Llama-2-7b-evolcodealpaca",
    )

    fig, axs = plt.subplots(
        4, figsize=(15, 10), sharex=True
    )  # Adjusting figsize for better layout

    plot_iter_tokens_hist(prom, args.time_window_expr, ax=axs[0])
    axs[0].set_title("Iteration Tokens Histogram")

    plot_iter_tokens_hist(prom, args.time_window_expr, ax=axs[1], normalize=True)
    axs[1].set_title("Iteration Tokens Histogram (pct)")

    plot_iter_tpot_hist(prom, args.time_window_expr, ax=axs[2])
    axs[2].set_title("Time per Output Token Histogram")

    plot_iter_tokens_sum_count(prom, args.time_window_expr, ax=axs[3])
    axs[3].set_title("Total Tokens and Iterations Over Time")

    for ax in axs:
        hide_every_n_xticks(ax, 2)

    plt.tight_layout()  # Adjust subplots to fit into figure area.
    plt.savefig("test.pdf", bbox_inches="tight")
