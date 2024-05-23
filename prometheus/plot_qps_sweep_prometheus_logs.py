import json
import argparse
import requests
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


class PrometheusClient:
    def __init__(self, server_url):
        self.server_url = server_url
        self.query_api_url = f"{self.server_url}/api/v1/query"

    def get_metric(self, metric_name, time_window_expr, model_id):
        query_str = (
            f'increase({metric_name}{{model_name="{model_id}"}}{time_window_expr})'
        )
        response = requests.get(self.query_api_url, params={"query": query_str})
        return response.json()["data"]["result"]


class BenchmarkPlotter:
    def __init__(self, results_file, prometheus_client):
        with open(results_file, "r") as file:
            self.results = json.load(file)
        self.qps_sweep = self.results["qps_sweep"]
        self.prom = prometheus_client
        self.model_id = self.results["model_id"]

    def plot_metrics(self):
        fig, axs = plt.subplots(3, figsize=(10, 10), sharex=True)
        fig.suptitle("Benchmark Metrics")

        self.plot_iteration_tokens_hist(ax=axs[0], normalize=True)
        axs[0].set_title("Iteration Tokens Histogram (pct)")

        self.plot_time_per_output_token_hist(ax=axs[1], normalize=True)
        axs[1].set_title("Time per Output Token Histogram (pct)")

        self.plot_time_to_first_token_hist(ax=axs[2], normalize=True)
        axs[2].set_title("Time to First Token Histogram (pct)")

        axs[-1].set_xlabel("QPS")

        plt.tight_layout()
        plt.savefig("benchmark_metrics.pdf", bbox_inches="tight")
        plt.show()

    def plot_hist(self, metric_name, include_inf=True, ax=None, normalize=False):
        data = {}
        for result in self.qps_sweep:
            promql_time = result["promql_window"]
            qps = result["qps"]
            metric_results = self.prom.get_metric(
                metric_name, promql_time, self.model_id
            )

            for metric_result in metric_results:
                le = metric_result["metric"]["le"]
                if le not in data:
                    data[le] = {}
                data[le][qps] = float(metric_result["value"][1])

        df = pd.DataFrame.from_dict(data)

        if not include_inf and "+Inf" in df.columns:
            df = df.drop(columns=["+Inf"])

        df ["0.0"] = 0 # add 0 so we can `diff` the first column with it
        df = df.reindex(
            sorted(df.columns, key=lambda x: int(float(x)) if x != "+Inf" else 1e39),
            axis=1,
        )
        df = df.diff(axis=1)
        df = df.drop(columns=["0.0"]) # remove 0, NANs after `diff(axis=1)`

        if normalize:
            df = df.div(df.sum(axis=1), axis=0)

        colors = plt.cm.GnBu(np.linspace(0, 1, df.shape[1]))

        df.plot.bar(stacked=True, color=colors, ax=ax)
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize="small")

    def plot_iteration_tokens_hist(self, ax=None, **kwargs):
        self.plot_hist(
            "vllm:iteration_tokens_total_bucket", include_inf=True, ax=ax, **kwargs
        )

    def plot_time_per_output_token_hist(self, ax=None, **kwargs):
        self.plot_hist(
            "vllm:time_per_output_token_seconds_bucket",
            include_inf=True,
            ax=ax,
            **kwargs,
        )

    def plot_time_to_first_token_hist(self, ax=None, **kwargs):
        self.plot_hist(
            "vllm:time_to_first_token_seconds_bucket", include_inf=True, ax=ax, **kwargs
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate benchmark plots from results."
    )
    parser.add_argument("results_file", type=str, help="Path to the JSON results file.")
    parser.add_argument(
        "--prometheus_url",
        type=str,
        help="Prometheus server URL.",
        default="http://localhost:9090",
    )
    args = parser.parse_args()

    prom_client = PrometheusClient(server_url=args.prometheus_url)
    plotter = BenchmarkPlotter(args.results_file, prom_client)
    plotter.plot_metrics()
