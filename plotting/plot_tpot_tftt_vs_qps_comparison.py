import json
import argparse
import matplotlib.pyplot as plt
import numpy as np

def load_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def plot_comparison(data_list, labels, output_file):
    num_files = len(data_list)
    bar_width = 0.8 / num_files
    index = np.arange(len(data_list[0]['qps_sweep']))

    plt.figure(figsize=(12, 8))

    for i, (data, label) in enumerate(zip(data_list, labels)):
        qps = [entry['qps'] for entry in data['qps_sweep']]
        mean_tpot_ms = [entry['mean_tpot_ms'] for entry in data['qps_sweep']]
        p99_tpot_ms = [entry['p99_tpot_ms'] for entry in data['qps_sweep']]
        mean_ttft_ms = [entry['mean_ttft_ms'] for entry in data['qps_sweep']]
        p99_ttft_ms = [entry['p99_ttft_ms'] for entry in data['qps_sweep']]

        plt.subplot(2, 2, 1)
        plt.bar(index + i * bar_width, mean_tpot_ms, bar_width, label=label)
        plt.xlabel('QPS')
        plt.ylabel('Mean TPOT (ms)')
        plt.yscale('log')
        plt.xticks(index + bar_width * (num_files - 1) / 2, qps)
        plt.title('Mean TPOT vs QPS')

        plt.subplot(2, 2, 2)
        plt.bar(index + i * bar_width, p99_tpot_ms, bar_width, label=label)
        plt.xlabel('QPS')
        plt.ylabel('P99 TPOT (ms)')
        plt.yscale('log')
        plt.xticks(index + bar_width * (num_files - 1) / 2, qps)
        plt.title('P99 TPOT vs QPS')

        plt.subplot(2, 2, 3)
        plt.bar(index + i * bar_width, mean_ttft_ms, bar_width, label=label)
        plt.xlabel('QPS')
        plt.ylabel('Mean TTFT (ms)')
        plt.yscale('log')
        plt.xticks(index + bar_width * (num_files - 1) / 2, qps)
        plt.title('Mean TTFT vs QPS')

        plt.subplot(2, 2, 4)
        plt.bar(index + i * bar_width, p99_ttft_ms, bar_width, label=label)
        plt.xlabel('QPS')
        plt.ylabel('P99 TTFT (ms)')
        plt.yscale('log')
        plt.xticks(index + bar_width * (num_files - 1) / 2, qps)
        plt.title('P99 TTFT vs QPS')

    plt.subplot(2, 2, 1)
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Plot saved as {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot throughput metrics comparison from multiple JSON files.')
    parser.add_argument('file_paths', nargs='+', type=str, help='Paths to the JSON files containing throughput data.')
    parser.add_argument('--labels', nargs='*', type=str, default=None, help='Labels for the JSON files data.')
    parser.add_argument('--output', type=str, default='ttft_tpot_vs_qps_comparison.pdf', help='Output PDF file name')
    
    args = parser.parse_args()

    file_paths = args.file_paths
    labels = args.labels if args.labels else file_paths

    if len(labels) != len(file_paths):
        print("Error: The number of labels must match the number of file paths.")
        exit(1)

    data_list = [load_json(file_path) for file_path in file_paths]

    plot_comparison(data_list, labels, args.output)
