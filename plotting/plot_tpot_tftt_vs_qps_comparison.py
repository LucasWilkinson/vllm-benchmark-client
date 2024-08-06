import json
import argparse
import matplotlib.pyplot as plt

def load_json(file_paths):
    merged_data = None
    for file_path in file_paths:
        with open(file_path, 'r') as file:
            data = json.load(file)
            
        if merged_data is not None:
            merged_data["qps_sweep"].extend(data["qps_sweep"])
        else:
            merged_data = data
        merged_data["qps_sweep"] = sorted(merged_data["qps_sweep"], 
                                          key=lambda x: x['qps'])
    return merged_data

def plot_throughput(data_list, output_file):
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    for name, data in data_list:
        qps = [entry['qps'] for entry in data['qps_sweep']]
        mean_tpot_ms = [entry['mean_tpot_ms'] for entry in data['qps_sweep']]
        plt.plot(qps, mean_tpot_ms, marker='o', linestyle='-', label=name)
    plt.xlabel('QPS')
    plt.ylabel('Mean TPOT (ms)')
    plt.title('Mean TPOT vs QPS')

    plt.subplot(2, 2, 2)
    for name, data in data_list:
        qps = [entry['qps'] for entry in data['qps_sweep']]
        p99_tpot_ms = [entry['p99_tpot_ms'] for entry in data['qps_sweep']]
        plt.plot(qps, p99_tpot_ms, marker='o', linestyle='-', label=name)
    plt.xlabel('QPS')
    plt.ylabel('P99 TPOT (ms)')
    plt.title('P99 TPOT vs QPS')

    plt.subplot(2, 2, 3)
    for name, data in data_list:
        qps = [entry['qps'] for entry in data['qps_sweep']]
        mean_ttft_ms = [entry['mean_ttft_ms'] for entry in data['qps_sweep']]
        plt.plot(qps, mean_ttft_ms, marker='o', linestyle='-', label=name)
    plt.xlabel('QPS')
    plt.ylabel('Mean TTFT (ms)')
    plt.title('Mean TTFT vs QPS')

    plt.subplot(2, 2, 4)
    for name, data in data_list:
        qps = [entry['qps'] for entry in data['qps_sweep']]
        p99_ttft_ms = [entry['p99_ttft_ms'] for entry in data['qps_sweep']]
        plt.plot(qps, p99_ttft_ms, marker='o', linestyle='-', label=name)
    plt.xlabel('QPS')
    plt.ylabel('P99 TTFT (ms)')
    plt.title('P99 TTFT vs QPS')

    handles, labels = plt.gca().get_legend_handles_labels()
    plt.gcf().suptitle(" vs. ".join([name for name, _ in data_list]))
    plt.gcf().legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.96), 
                     ncol=len(labels))
    plt.tight_layout()
    plt.savefig(output_file, bbox_inches='tight')
    print(f"Plot saved as {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot throughput metrics from JSON file.')
    parser.add_argument('--src', nargs='+', type=str, 
                        help='Path to the JSON file containing throughput data.', action='append')
    parser.add_argument('--output', type=str, default='ttft_tpot_vs_qps.pdf', help='Output PDF file name')
    
    args = parser.parse_args()
    print(args.src)
    
    data_list = [(s[0], load_json(s[1:])) for s in args.src]
    plot_throughput(data_list, args.output)
