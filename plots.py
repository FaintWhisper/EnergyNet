import matplotlib as mpl
# change font family to 'Liberation Serif'
mpl.rc('font',family='Liberation Serif')
from matplotlib.patches import FancyBboxPatch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path

def plot_stats(experiment, device, test):
    df_test = pd.read_csv(f"poc_energy_efficiency_crypto/crypto_spider_5g_fcnn_optimized_benchmark_{test}_stats.csv")
    
    print(df_test)
    
    stats_list = df_test.columns[3:]
    # stats_names = [stat.replace("_", " ").capitalize() for stat in stats_list]
    stats_names = {stat: " ".join(stat.split("_")).capitalize() for stat in stats_list}
        
    for stat in stats_list:
        print(f"Plotting {stat} for {experiment} on {device} for {test}")
        fig, ax = plt.subplots(figsize=(8, 5))

        sns.set(style="darkgrid")
        sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})
        sns.set_style("whitegrid", {"axes.grid": False})

        stat_data = df_test[(df_test["experiment"] == experiment) & (df_test["device"] == device)][stat]
        baseline_stat_data = df_test[(df_test["experiment"] == "EXP0") & (df_test["device"] == device)][stat]
        
        if experiments[experiment]["post_training_optimizations"] is not None:
            post_training_optimizations = " ".join(experiments[experiment]["post_training_optimizations"]).replace("_", " ").capitalize()
        else:
            post_training_optimizations = ""
        if experiments[experiment]["training_aware_optimizations"] is not None:
            training_aware_optimizations = " ".join(experiments[experiment]["training_aware_optimizations"]).replace("_", " ").capitalize()
        else:
            training_aware_optimizations = ""
        
        experiment_name = f"{post_training_optimizations} {training_aware_optimizations}"
        print(experiment_name)
        
        time = np.arange(0, len(stat_data))
        baseline_time = np.arange(0, len(baseline_stat_data))
        
        max_time = max(len(time), len(baseline_time))
        
        ax = sns.lineplot(x=time, y=stat_data, label=experiment_name)
        ax = sns.lineplot(x=baseline_time, y=baseline_stat_data, label="Baseline")

        ax.set(xlabel="Time (s)", ylabel=stat)
        ax.set(ylabel=stats_names[stat])
        # ax.set_xticks(np.arange(0, max_time + 1, 1))
        # ax.set_xticklabels(np.arange(0, max_time + 1, 1))
        ax.set_xlim(0, max(time))
        ax.set_ylim(0, max(df_test[stat]) * 1.1)
        # ax.set_title(f"{stats_names[stat]}. {experiment_name} ({device})", fontsize=17)
        ax.set_title(f"{stats_names[stat]}. {experiment} ({device})", fontsize=17)
        ax.legend(prop={'family':'Liberation Serif'}, loc='upper left')
        plt.tight_layout()
        # plt.show()
        Path(f"poc_energy_efficiency_crypto/plots").mkdir(parents=True, exist_ok=True)
        plt.savefig(f"poc_energy_efficiency_crypto/plots/crypto_spider_5g_fcnn_optimized_benchmark_{test}_stats_{experiment}_{device}_{stat}.png", dpi=300)
        plt.close()

# plot df_inference_stats for all batch sizes
def plot_inference_stats(device, df_inference_stats, batch_size):
    print(df_inference_stats)
    
    stat = "Percentage of Total Avg. CPU Energy Consumption Reduction (%)"
    stat_title = "Energy Consumption Reduction (%)"
    
    print(f"Plotting {stat} ({device}) for batch size {batch_size}")
    fig, ax = plt.subplots(figsize=(8, 8))

    # sns.set(style="darkgrid", font_scale=1.5)
    # sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})
    # sns.set_style("whitegrid", {"axes.grid": False})
    sns.set_style("darkgrid", {"axes.grid": True, "font.family": "Liberation Serif"})
    
    # change color of the background of the grid
    sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5, "axes.facecolor": (0, 0, 0, 0)})

    exps = np.arange(1, len(df_inference_stats[stat]))
    
    # exps should be the x-axis and the bars should be the y-axis, that is, the bars should be arranged vertically
    ax = sns.barplot(x=exps, y=df_inference_stats[stat].values[1:], label=stat_title, palette="Set2", linewidth=0.6, edgecolor=".35", saturation=0.93, ci=None)
    
    # put the labels on the bars
    values = df_inference_stats[stat].values[1:]
    
    for i, v in enumerate(values):
        offset = 3 if v > 0 else -6
        ax.text(i, v + offset, str(round(v, 2)), color=(0.18, 0.18, 0.18), fontweight='bold', ha="center", fontfamily="Liberation Serif", fontsize=12)
       
    # set y-axis upper limit to 100
    ax.set_ylim(-115, 100)
    
    ax.set_yticks(np.arange(-125, 125, 25))

    # ax.set(xlabel="Time (s)", ylabel=stat)
    # ax.set(ylabel=stat.replace("_", " ").capitalize())
    # ax.set_xlim(0, max(time))
    # ax.set_ylim(0, max(df_inference_stats[stat]) * 1.1)
    ax.set_title(f"{stat_title}. Batch size {batch_size}", fontsize=15, fontfamily="Liberation Serif")
    # ax.legend(prop={'family':'Liberation Serif'}, loc="lower right")
    # change axis labels font family
    ax.set_xticklabels(ax.get_xticklabels(), fontfamily="Liberation Serif")
    ax.set_yticklabels(ax.get_yticklabels(), fontfamily="Liberation Serif")
    ax.set_xlabel("Optimization Strategies", fontfamily="Liberation Serif")
    ax.set_ylabel(stat_title, fontfamily="Liberation Serif")
 
    plt.tight_layout()
    # plt.show()
    Path(f"poc_energy_efficiency_crypto/plots").mkdir(parents=True, exist_ok=True)
    plt.savefig(f"poc_energy_efficiency_crypto/plots/crypto_spider_5g_fcnn_optimized_benchmark_inference_stats_batch_size_{batch_size}_{stat}.png", dpi=300)
    plt.close()

def plot_results():
    # plot results
    tests = ["training", "inference", "load"]

    for experiment in experiments:
        if experiment == "EXP0":
            continue
        
        for device in devices:
            for test in tests:
                # plot_stats(experiment, device, test)
                pass
    
    print("Plotting inference stats for all batch sizes")

    for device in devices:
        for batch_size in batch_sizes:
            df_inference_stats = pd.read_csv(f"results_final/rev6/{batch_size}/inference_stats_{batch_size}_final.csv")
            
            plot_inference_stats(device, df_inference_stats, batch_size)