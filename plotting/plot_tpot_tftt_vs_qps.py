import json
import argparse
import matplotlib.pyplot as plt

def load_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def plot_throughput(data, output_file):
    qps = [entry['qps'] for entry in data['qps_sweep']]
    mean_tpot_ms = [entry['mean_tpot_ms'] for entry in data['qps_sweep']]
    p99_tpot_ms = [entry['p99_tpot_ms'] for entry in data['qps_sweep']]
    mean_ttft_ms = [entry['mean_ttft_ms'] for entry in data['qps_sweep']]
    p99_ttft_ms = [entry['p99_ttft_ms'] for entry in data['qps_sweep']]

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.plot(qps, mean_tpot_ms, marker='o', linestyle='-', color='b')
    plt.xlabel('QPS')
    plt.ylabel('Mean TPOT (ms)')
    plt.title('Mean TPOT vs QPS')

    plt.subplot(2, 2, 2)
    plt.plot(qps, p99_tpot_ms, marker='o', linestyle='-', color='g')
    plt.xlabel('QPS')
    plt.ylabel('P99 TPOT (ms)')
    plt.title('P99 TPOT vs QPS')

    plt.subplot(2, 2, 3)
    plt.plot(qps, mean_ttft_ms, marker='o', linestyle='-', color='r')
    plt.xlabel('QPS')
    plt.ylabel('Mean TTFT (ms)')
    plt.title('Mean TTFT vs QPS')

    plt.subplot(2, 2, 4)
    plt.plot(qps, p99_ttft_ms, marker='o', linestyle='-', color='c')
    plt.xlabel('QPS')
    plt.ylabel('P99 TTFT (ms)')
    plt.title('P99 TTFT vs QPS')

    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Plot saved as {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot throughput metrics from JSON file.')
    parser.add_argument('file_path', type=str, help='Path to the JSON file containing throughput data.')
    parser.add_argument('--output', type=str, default='ttft_tpot_vs_qps.pdf', help='Output PDF file name')
    
    args = parser.parse_args()
    
    data = load_json(args.file_path)
    plot_throughput(data, args.output)
