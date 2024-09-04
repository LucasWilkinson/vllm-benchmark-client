import math
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.colors as mcolors

def plot_hist_df(df,
                 ax,
                 scale_y_factor=1.0,
                 xlabels_rotation=0,
                 buckets=None):
    df = df.drop('metric', axis=1)
    plot_df = df

    if buckets:
        for bucket in buckets:
            bucket_start, bucket_end = bucket[0].split("-")
            if not any(f"{bucket_start}-" in b for b in df.columns):
                raise ValueError(
                    f"Bucket start {bucket_start} not found in the data.")
            if not any(f"-{bucket_end}" in b for b in df.columns):
                raise ValueError(
                    f"Bucket end {bucket_end} not found in the data.")
        plot_df = pd.DataFrame(index=df.index.copy(), 
                                columns=[bucket[1] for bucket in buckets])
        for col in plot_df.columns:
            plot_df[col].values[:] = 0
            
        def bucket_start_end(s):
            return [int(x) if x != "+Inf" else math.inf 
                    for x in s.split("-")]
            
        for src_col in df.columns:
            src_bucket_start, src_bucket_end = bucket_start_end(src_col)
            for bucket in buckets:
                bucket_start, bucket_end = bucket_start_end(bucket[0])
                if src_bucket_start >= bucket_start and src_bucket_end <= bucket_end:
                    plot_df[bucket[1]] += df[src_col]

    num_qps = len(plot_df.index)
    cmap = plt.get_cmap("viridis")
    colors = [cmap(i / num_qps) for i in range(num_qps)]
    
    for i, (qps, row) in enumerate(plot_df.iterrows()):
        # Plot the QPS value as a bar with gradient color and rounded edges
        bar_width = 0.8 / num_qps
        x = np.arange(len(plot_df.columns))

        ax.bar(x + i * bar_width, 
               row * scale_y_factor, 
               bar_width, 
               label=f'{qps}', 
               color=colors[i], 
               edgecolor='black', 
               linewidth=0.5, 
               zorder=2)

        ax.set_xticks(x + bar_width * (num_qps - 1) / 2)
        ax.set_xticklabels(list(plot_df.columns), rotation=xlabels_rotation)
        ax.grid(True, zorder=1)  # Add grid behind bars
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot the output of create_csv_for_hist_metric.py")
    parser.add_argument("csvs", type=str, 
                        help="Path to the CSV result files to plot.", nargs='+')
    parser.add_argument(
        "--outfile",
        type=str,
        help="Path to save the output plot.",
        default="csv_histogram.pdf",
    )
    parser.add_argument(
        "--buckets",
        type=str,
        help="Buckets to group the data into. Format: 'start-end:label'.",
        nargs='+',
    )
    parser.add_argument(
        "--titles",
        type=str,
        help="Titles for the plots.",
        nargs='+',
    )
    parser.add_argument(
        "--sup-title",
        type=str,
        help="Super title for the plot.",
        default=None,
    )
    parser.add_argument(
        "--share-x",
        action="store_true",
        help="Share the x-axis between plots.",
    )
    parser.add_argument(
        "--xlabel",
        type=str,
        help="X-axis label.",
        default=None,
    )
    parser.add_argument(
        "--xlabels-rotation",
        type=int,
        help="X-axis label rotation.",
        default=0,
    )
    parser.add_argument(
        "--ylabel",
        type=str,
        help="Y-axis label.",
        default=None,
    )
    parser.add_argument(
        "--scale-y-factor",
        type=float,
        help="Factor to scale the y-axis by.",
        default=1.0,
    )
    parser.add_argument(
        "--subplot-height",
        type=float,
        help="Height of plot = subplot_height * number of CSVs.",
        default=3.3,
    )
    parser.add_argument(
        "--plot-width", 
        type=float,
        help="Width of plot.",
        default=10.0,
    )
    args = parser.parse_args()
    
    fig, axs = plt.subplots(len(args.csvs), 1, 
                            figsize=(args.plot_width, 
                                     args.subplot_height * len(args.csvs)), 
                            squeeze=False,
                            sharex=args.share_x)
    
    if args.sup_title:
        fig.suptitle(args.sup_title)
    
    if args.titles:
        assert len(args.titles) == len(args.csvs)

    for i, csv in enumerate(args.csvs):
        df = pd.read_csv(csv, index_col=0)
        plot_hist_df(df, 
                     axs.flatten()[i],
                     scale_y_factor=args.scale_y_factor,
                     xlabels_rotation=args.xlabels_rotation,
                     buckets=[
                         b.split(":") for b in args.buckets] 
                     if args.buckets else None)
        
        if not args.titles:
            axs[i, 0].set_title(csv)
        else:
            axs[i, 0].set_title(args.titles[i])

        axs[i, 0].legend(title="QPS")

        if args.xlabel and (not args.share_x or i == len(args.csvs) - 1):
            axs[i, 0].set_xlabel(args.xlabel)
        if args.ylabel:
            axs[i, 0].set_ylabel(args.ylabel)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(args.outfile, bbox_inches='tight')
    print("Saved", args.outfile)
    