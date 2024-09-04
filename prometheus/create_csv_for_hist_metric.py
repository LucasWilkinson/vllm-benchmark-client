import json
import argparse
import requests
import pandas as pd

from typing import List

class PrometheusClient:
    def __init__(self, server_url):
        self.server_url = server_url
        self.query_api_url = f'{self.server_url}/api/v1/query'

    def get_metric(self, metric_name, time_window_expr, model_id):
        query_str = f"increase({metric_name}{{model_name=\"{model_id}\"}}{time_window_expr})"
        response = requests.get(self.query_api_url, params={'query': query_str})
        return response.json()['data']['result']

def create_df(prom: PrometheusClient,
                model_id: str, 
                metric_name: str, 
                qps: List[int], 
                promql_time: str, 
                include_inf=True, 
                normalize=True):
    data = {}
    metric_results = prom.get_metric(
        metric_name, promql_time, model_id)

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

    df["metric"] = metric_name
    return df

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
        default="prom_results.csv",
    )
    parser.add_argument(
        "--metric",
        type=str,
        help="Metric to create the CSV for.",
        default="http://localhost:9090",
    )
    args = parser.parse_args()

    prom_client = PrometheusClient(server_url=args.prometheus_url)
    
    with open(args.results_file, 'r') as file:
        results = json.load(file)
    qps_sweep = results['qps_sweep']
    model_id = results['model_id']
    outfile = args.outfile
    
    df = None
    for result in qps_sweep:
        _df = create_df(prom_client, 
                        model_id, 
                        "vllm:iteration_tokens_total_bucket", 
                        result['qps'], 
                        result['promql_window'])
        
        if df is None:
            df = _df
        else:
            df = pd.concat([df, _df], axis=0)
    outfile = outfile.replace("%", args.results_file.replace(".json", ""))
    print(df)
    df.to_csv(outfile)
    print("Saved: ", outfile)
