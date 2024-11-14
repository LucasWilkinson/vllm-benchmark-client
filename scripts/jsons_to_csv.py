import json
import glob
import argparse
import matplotlib.pyplot as plt
import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Convert metrics from JSON file to CSV.')
    parser.add_argument('pats',
                        nargs='+',
                        type=str,
                        help='Glob patterns for CSV')

    args = parser.parse_args()
    rows = []
    for pat in args.pats:
        for file_path in glob.glob(pat):
            print("loading", file_path)
            with open(file_path, 'r') as file:
                data = json.load(file)
            for entry in data['qps_sweep']:
                row = entry.copy()
                row['file'] = file_path
                row['dataset'] = data['dataset']
                row['model_id'] = data['model_id']
                rows.append(row)

    pd.DataFrame(rows).to_csv('metrics.csv', index=False)
