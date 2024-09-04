import json
import math
import argparse
import requests
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.colors as mcolors

from typing import List, Tuple

class PrometheusClient:
    def __init__(self, server_url):
        self.server_url = server_url
        self.query_api_url = f'{self.server_url}/api/v1/query'

    def get_metric(self, metric_name, time_window_expr, model_id):
        query_str = f"increase({metric_name}{{model_name=\"{model_id}\"}}{time_window_expr})"
        response = requests.get(self.query_api_url, params={'query': query_str})
        return response.json()['data']['result']

class BenchmarkPlotter:
    def __init__(self, results_file, prometheus_client, outfile):
        with open(results_file, 'r') as file:
            self.results = json.load(file)
        self.qps_sweep = self.results['qps_sweep']
        self.prom = prometheus_client
        self.model_id = self.results['model_id']
        self.outfile = outfile

    def plot_metrics(self):
        metrics = [
            ("Iteration Tokens Histogram by QPS", "vllm:iteration_tokens_total_bucket"),
        ]
        
        fig, axs = plt.subplots(len(metrics), 1, 
                                figsize=(10, 3.3 * len(metrics)), squeeze=False)

        for ax, (title, metric_name) in zip(axs.flatten(), metrics):
            self.plot_metric(ax=ax, 
                             metric_name=metric_name, 
                             title=title)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(self.outfile, bbox_inches='tight')
        print("Saved", self.outfile)

    def plot_metric(self, ax, metric_name, title):
        
        
        ax.set_title(title)
        cmap = plt.get_cmap("viridis")
        num_qps = len(self.qps_sweep)
        colors = [cmap(i / num_qps) for i in range(num_qps)]

        for idx, result in enumerate(self.qps_sweep):
            self.plot_hist(ax=ax, 
                           metric_name=metric_name, 
                           qps=result['qps'], 
                           promql_time=result['promql_window'], 
                           normalize=True, 
                           idx=idx, 
                           color=colors[idx],
                           buckets=[("0-128", "Memory Bound (0-128)"), 
                                    ("128-+Inf", "Compute Bound (128+)")])
        
        #ax.set_xlabel("Buckets")
        ax.legend(title="QPS")

    def plot_hist(self, metric_name, qps, promql_time, include_inf=True, 
                  ax=None, normalize=True, idx=0, color=None, 
                  buckets: List[Tuple[str, str]]=None):
        data = {}
        metric_results = self.prom.get_metric(
            metric_name, promql_time, self.model_id)

        for metric_result in metric_results:
            le = metric_result['metric']['le']
            if le not in data:
                data[le] = {}
            data[le][qps] = float(metric_result['value'][1])
                
        df = pd.DataFrame.from_dict(data)

        if not include_inf and "+Inf" in df.columns:
            df = df.drop(columns=["+Inf"])

        df["0.0"] = 0  # add 0 so we can `diff` the first column with it
        df = df.reindex(
            sorted(df.columns, key=lambda x: int(float(x)) if x != "+Inf" else 1e39), axis=1)
        df = df.diff(axis=1)
        df = df.drop(columns=["0.0"])  # remove 0, NANs after `diff(axis=1)`

        df = df.rename(columns=lambda x: x.replace(".0", ""))
        new_columns = df.columns[:-1].str.cat("-" + df.columns[1:])
        new_columns = new_columns.insert(0, "0-" + df.columns[0])
        df = df.rename(columns=dict(zip(df.columns, new_columns)))

        if normalize:
            df = df.div(df.sum(axis=1), axis=0)

        # Plot the QPS value as a bar with gradient color and rounded edges
        num_qps = len(self.qps_sweep)
        bar_width = 0.8 / num_qps
        x = np.arange(len(df.columns))

        ax.bar(x + idx * bar_width, 
               df.loc[qps], 
               bar_width, 
               label=f'{qps}', 
               color=color, 
               edgecolor='black', 
               linewidth=0.5, 
               zorder=2)

        ax.set_xticks(x + bar_width * (num_qps - 1) / 2)
        ax.set_xticklabels(list(df.columns))
        ax.grid(True, zorder=1)  # Add grid behind bars

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate benchmark plots from results.")
    parser.add_argument("results_file", type=str, help="Path to the JSON results file.")
    parser.add_argument(
        "--prometheus_url",
        type=str,
        help="Prometheus server URL.",
        default="http://localhost:9090",
    )
    parser.add_argument(
        "--outfile",
        type=str,
        help="Path to save the output plot.",
        default="iteration_tokens_histogram.pdf",
    )
    args = parser.parse_args()

    prom_client = PrometheusClient(server_url=args.prometheus_url)
    plotter = BenchmarkPlotter(args.results_file, prom_client, args.outfile)
    plotter.plot_metrics()
