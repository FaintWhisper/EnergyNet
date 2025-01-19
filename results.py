import pandas as pd
import os
from pathlib import Path
from tabulate import tabulate

def clean_files():
    if os.path.exists(f"poc_energy_efficiency_crypto/crypto_spider_5g_fcnn_optimized_benchmark_times.csv"):
        os.remove(f"poc_energy_efficiency_crypto/crypto_spider_5g_fcnn_optimized_benchmark_times.csv")

    if os.path.exists("poc_energy_efficiency_crypto/crypto_spider_5g_fcnn_optimized_benchmark_training_stats.csv"):
        os.remove("poc_energy_efficiency_crypto/crypto_spider_5g_fcnn_optimized_benchmark_training_stats.csv")

    if os.path.exists("poc_energy_efficiency_crypto/crypto_spider_5g_fcnn_optimized_benchmark_inference_stats.csv"):
        os.remove("poc_energy_efficiency_crypto/crypto_spider_5g_fcnn_optimized_benchmark_inference_stats.csv")

    if os.path.exists("poc_energy_efficiency_crypto/crypto_spider_5g_fcnn_optimized_benchmark_load_stats.csv"):
        os.remove("poc_energy_efficiency_crypto/crypto_spider_5g_fcnn_optimized_benchmark_load_stats.csv")

    if os.path.exists("poc_energy_efficiency_crypto/crypto_spider_5g_fcnn_optimized_benchmark_evaluation.csv"):
        os.remove("poc_energy_efficiency_crypto/crypto_spider_5g_fcnn_optimized_benchmark_evaluation.csv")

def load_files():
    df_times = pd.read_csv(f"poc_energy_efficiency_crypto/crypto_spider_5g_fcnn_optimized_benchmark_times.csv")
    df_training_stats = pd.read_csv(f"poc_energy_efficiency_crypto/crypto_spider_5g_fcnn_optimized_benchmark_training_stats.csv")
    df_inference_stats = pd.read_csv(f"poc_energy_efficiency_crypto/crypto_spider_5g_fcnn_optimized_benchmark_inference_stats.csv")
    df_load_stats = pd.read_csv(f"poc_energy_efficiency_crypto/crypto_spider_5g_fcnn_optimized_benchmark_load_stats.csv")
    df_evaluation = pd.read_csv(f"poc_energy_efficiency_crypto/crypto_spider_5g_fcnn_optimized_benchmark_evaluation.csv")
    
    return df_times, df_training_stats, df_inference_stats, df_load_stats, df_evaluation

def aggregate_stats_by_time(df):
    columns_avg_std = [col for col in df.columns if "max" not in col]
    
    df_avg_std = df[columns_avg_std].groupby(["experiment", "device"]).mean().reset_index()
    df_avg_std = df_avg_std.drop(columns=["snapshot"])
    
    # columns containing "max" in the name must be aggregated by taking the maximum value instead of the mean
    columns_max = [col for col in df.columns if "max" in col]
    
    # add experiment and device columns to columns_max
    columns_max.append("experiment")
    columns_max.append("device")
    
    df_max = df[columns_max]
    df_max = df_max.groupby(["experiment", "device"]).max().reset_index()
    
    df_final = pd.concat([df_avg_std, df_max[columns_max[:-2]]], axis=1)
    
    return df_final

def sort_by_experiment(df):    
    # sort dataframes by experiment. experiment is the string "EXP" followed by a number, so we can sort by the number
    df['sort'] = df['experiment'].str.extract('(\d+)', expand=False).astype(int)
    df.sort_values('sort',inplace=True, ascending=True)
    df = df.drop('sort', axis=1)
    df = df.reset_index(drop=True)
    
    return df

def calculate_energy_consumption(df_experiment, experiment, df_times):    
    average_exp_duration = df_times[f"average_{experiment}_time"]
    df_experiment["total_average_cpu_energy_consumption"] = df_experiment[f"average_{experiment}_cpu_power_draw"] * average_exp_duration
    df_experiment["total_average_gpu_energy_consumption"] = df_experiment[f"average_{experiment}_gpu_power_draw"] * average_exp_duration
    
    # calculate percentage of energy consumption reduction with respect to the baseline (EXP0)
    baseline = df_experiment[df_experiment["experiment"] == "EXP0"]
    baseline_cpu_energy_consumption = baseline["total_average_cpu_energy_consumption"].values[0]
    baseline_gpu_energy_consumption = baseline["total_average_gpu_energy_consumption"].values[0]
    
    percentage_cpu_energy_consumption_reduction = (df_experiment["total_average_cpu_energy_consumption"] - baseline_cpu_energy_consumption) / baseline_cpu_energy_consumption * 100
    percentage_gpu_energy_consumption_reduction = (df_experiment["total_average_gpu_energy_consumption"] - baseline_gpu_energy_consumption) / baseline_gpu_energy_consumption * 100
    
    # round to 5 decimal places
    percentage_cpu_energy_consumption_reduction = percentage_cpu_energy_consumption_reduction.round(5)
    percentage_gpu_energy_consumption_reduction = percentage_gpu_energy_consumption_reduction.round(5)
    
    # if number is positive, put a "+" in front of it
    # percentage_cpu_energy_consumption_reduction = percentage_cpu_energy_consumption_reduction.apply(lambda x: f"+{x}" if x > 0 else x)
    # percentage_gpu_energy_consumption_reduction = percentage_gpu_energy_consumption_reduction.apply(lambda x: f"+{x}" if x > 0 else x)
        
    df_experiment["percentage_cpu_energy_consumption_reduction"] = percentage_cpu_energy_consumption_reduction
    df_experiment["percentage_gpu_energy_consumption_reduction"] = percentage_gpu_energy_consumption_reduction
    
    # in baseline put a "N/A" in the percentage column
    df_experiment.loc[df_experiment["experiment"] == "EXP0", "percentage_cpu_energy_consumption_reduction"] = "N/A"
    df_experiment.loc[df_experiment["experiment"] == "EXP0", "percentage_gpu_energy_consumption_reduction"] = "N/A"
        
    return df_experiment

def sort_columns(df_training_stats, df_inference_stats, df_load_stats):
    print("Sorting columns...")
    
    stats = ["ram_memory_uss", "ram_memory_rss", "ram_memory_vms", "ram_memory_pss", "cpu_usage", "cpu_freq", "cpu_power_draw", "io_usage", "bytes_written", "bytes_read", "gpu_memory", "gpu_usage", "gpu_freq", "gpu_power_draw"]
    tests = ["training", "inference", "load"]
    
    for test in tests:
        test_columns = ["experiment", "device"]
        
        for i in range(len(stats)):
            stat_test_columns = [f"average_{test}_{stats[i]}", f"std_dev_{test}_{stats[i]}", f"max_{test}_{stats[i]}"]
            test_columns.extend(stat_test_columns)
        
        test_columns.extend(["total_average_cpu_energy_consumption", "percentage_cpu_energy_consumption_reduction", "total_average_gpu_energy_consumption", "percentage_gpu_energy_consumption_reduction"])
        
        if test == "training":
            df_training_stats = df_training_stats[test_columns]
        elif test == "inference":
            df_inference_stats = df_inference_stats[test_columns]
        elif test == "load":
            df_load_stats = df_load_stats[test_columns]
    
    return df_training_stats, df_inference_stats, df_load_stats

def move_model_size(df_times, df_load_stats):
    print("Moving 'model size' column from times to load stats dataframe...")
    
    df_load_stats["model_size"] = df_times["model_size"]
    df_times = df_times.drop("model_size", axis=1)
    
    return df_times, df_load_stats

def pretty_format_column_names(df_times, df_training_stats, df_inference_stats, df_load_stats, df_evaluation):
    print("Pretty formating column names...")
    
    stats = ["ram_memory_uss", "ram_memory_rss", "ram_memory_vms", "ram_memory_pss", "cpu_usage", "cpu_freq", "cpu_power_draw", "io_usage", "bytes_written", "bytes_read", "gpu_memory", "gpu_usage", "gpu_freq", "gpu_power_draw"]

    stats_names = {
        "ram_memory_uss": "RAM Memory USS (B)",
        "ram_memory_rss": "RAM Memory RSS (B)",
        "ram_memory_vms": "RAM Memory VMS (B)",
        "ram_memory_pss": "RAM Memory PSS (B)",
        "cpu_usage": "CPU Usage (%)",
        "cpu_freq": "CPU Frequency (MHz)",
        "cpu_power_draw": "CPU Power Draw (W)",
        "io_usage": "I/O Usage (%)",
        "bytes_written": "Bytes Written (B)",
        "bytes_read": "Bytes Read (B)",
        "gpu_memory": "GPU Memory (MB)",
        "gpu_usage": "GPU Usage (%)",
        "gpu_freq": "GPU Frequency (MHz)",
        "gpu_power_draw": "GPU Power Draw (W)"
    }
    
    agg_types = ["Avg.", "Std. Dev.", "Max."]
    
    stats_column_names = ["Experiment", "Device"]
    
    for i in range(len(stats)):
        for j in range(len(agg_types)):
            stats_column_names.append(f"{agg_types[j]} {stats_names[stats[i]]}")
    
    stats_column_names.extend(["Total Avg. CPU Energy Consumption (J)", "Percentage of Total Avg. CPU Energy Consumption Reduction (%)", "Total Avg. GPU Energy Consumption (J)", "Percentage of Total Avg. GPU Energy Consumption Reduction (%)"])
            
    df_training_stats.columns = stats_column_names
    df_inference_stats.columns = stats_column_names
    df_load_stats.columns = stats_column_names + ["Model Size (B)"]
    
    binary_classification_evaluation_column_names = ["Experiment", "Device", "Accuracy", "F1 Score", "AUC", "Recall", "Precision", "Balanced Accuracy", "Matthews Correlation Coefficient"]
    regression_evaluation_column_names = ["Experiment", "Device", "Mean Squared Error", "Mean Absolute Error", "Mean Absolute Percentage Error", "Symmetric Mean Absolute Percentage Error"]
    
    evaluation_column_names = binary_classification_evaluation_column_names if ml_task == "binary_classification" else regression_evaluation_column_names
    
    df_evaluation.columns = evaluation_column_names
    
    times_names = {
        "training_time": "Training Time (s)",
        "inference_time": "Inference Time (s)",
        "load_time": "Load Time (s)",
    }
    
    times_column_names = ["Experiment", "Device"]
    
    for time_column in times_names:
        for agg_type in agg_types:
            times_column_names.append(f"{agg_type} {times_names[time_column]}")
    
    df_times.columns = times_column_names
            
    return df_times, df_training_stats, df_inference_stats, df_load_stats, df_evaluation

def round_results(df_times, df_training_stats, df_inference_stats, df_load_stats, df_evaluation, decimal_places=5):
    print("Rounding results...")
    
    df_times = df_times.round(decimal_places)
    df_training_stats = df_training_stats.round(decimal_places)
    df_inference_stats = df_inference_stats.round(decimal_places)
    df_load_stats = df_load_stats.round(decimal_places)
    df_evaluation = df_evaluation.round(decimal_places)
    
    return df_times, df_training_stats, df_inference_stats, df_load_stats, df_evaluation

def remove_unnecesary_stats(df_training_stats, df_inference_stats, df_load_stats):
    print("Removing unnecesary stats...")
    
    # if device is CPU, remove GPU columns
    df_training_stats = df_training_stats.drop(df_training_stats[df_training_stats["Device"] == "CPU"].filter(regex="GPU").columns, axis=1)
    df_inference_stats = df_inference_stats.drop(df_inference_stats[df_inference_stats["Device"] == "CPU"].filter(regex="GPU").columns, axis=1)
    df_load_stats = df_load_stats.drop(df_load_stats[df_load_stats["Device"] == "CPU"].filter(regex="GPU").columns, axis=1)
    
    # remove RSS columns
    df_training_stats = df_training_stats.drop(df_training_stats.filter(regex="RSS").columns, axis=1)
    df_inference_stats = df_inference_stats.drop(df_inference_stats.filter(regex="RSS").columns, axis=1)
    df_load_stats = df_load_stats.drop(df_load_stats.filter(regex="RSS").columns, axis=1)
    
    # remove VMS columns
    df_training_stats = df_training_stats.drop(df_training_stats.filter(regex="VMS").columns, axis=1)
    df_inference_stats = df_inference_stats.drop(df_inference_stats.filter(regex="VMS").columns, axis=1)
    df_load_stats = df_load_stats.drop(df_load_stats.filter(regex="VMS").columns, axis=1)
    
    # remove PSS columns
    df_training_stats = df_training_stats.drop(df_training_stats.filter(regex="PSS").columns, axis=1)
    df_inference_stats = df_inference_stats.drop(df_inference_stats.filter(regex="PSS").columns, axis=1)
    df_load_stats = df_load_stats.drop(df_load_stats.filter(regex="PSS").columns, axis=1)
    
    # remove bytes written and bytes read columns
    df_training_stats = df_training_stats.drop(df_training_stats.filter(regex="Bytes").columns, axis=1)
    df_inference_stats = df_inference_stats.drop(df_inference_stats.filter(regex="Bytes").columns, axis=1)
    df_load_stats = df_load_stats.drop(df_load_stats.filter(regex="Bytes").columns, axis=1)
    
    # remove IO usage column
    df_training_stats = df_training_stats.drop(df_training_stats.filter(regex="I/O").columns, axis=1)
    df_inference_stats = df_inference_stats.drop(df_inference_stats.filter(regex="I/O").columns, axis=1)
    df_load_stats = df_load_stats.drop(df_load_stats.filter(regex="I/O").columns, axis=1)
    
    return df_training_stats, df_inference_stats, df_load_stats

def reorder_columns(df_training_stats, df_inference_stats, df_load_stats):
    # reorder columns
    order = ["Experiment", "Device", "Total Avg. CPU Energy Consumption (J)", "Percentage of Total Avg. CPU Energy Consumption Reduction (%)", "Total Avg. GPU Energy Consumption (J)", "Percentage of Total Avg. GPU Energy Consumption Reduction (%)", "Avg. CPU Power Draw (W)", "Std. Dev. CPU Power Draw (W)", "Max. CPU Power Draw (W)", "Avg. GPU Power Draw (W)", "Std. Dev. GPU Power Draw (W)", "Max. GPU Power Draw (W)", "Avg. CPU Usage (%)", "Std. Dev. CPU Usage (%)", "Max. CPU Usage (%)", "Avg. GPU Usage (%)", "Std. Dev. GPU Usage (%)", "Max. GPU Usage (%)", "Avg. CPU Frequency (MHz)", "Std. Dev. CPU Frequency (MHz)", "Max. CPU Frequency (MHz)", "Avg. GPU Frequency (MHz)", "Std. Dev. GPU Frequency (MHz)", "Max. GPU Frequency (MHz)", "Avg. RAM Memory USS (B)", "Std. Dev. RAM Memory USS (B)", "Max. RAM Memory USS (B)", "Avg. RAM Memory PSS (B)", "Std. Dev. RAM Memory PSS (B)", "Max. RAM Memory PSS (B)", "Avg. RAM Memory RSS (B)", "Std. Dev. RAM Memory RSS (B)", "Max. RAM Memory RSS (B)", "Avg. RAM Memory VMS (B)", "Std. Dev. RAM Memory VMS (B)", "Max. RAM Memory VMS (B)", "Avg. I/O Usage (%)", "Std. Dev. I/O Usage (%)", "Max. I/O Usage (%)", "Avg. Bytes Written (B)", "Std. Dev. Bytes Written (B)", "Max. Bytes Written (B)", "Avg. Bytes Read (B)", "Std. Dev. Bytes Read (B)", "Max. Bytes Read (B)"]
    
    df_training_stats = df_training_stats[order]
    df_inference_stats = df_inference_stats[order]
    df_load_stats = df_load_stats[order + ["Model Size (B)"]]
    
    return df_training_stats, df_inference_stats, df_load_stats

def group_with_evaluation(df_stats, df_evaluation):
    # add evaluation metrics to inference stats
    df_stats["Accuracy"] = df_evaluation["Accuracy"]
    df_stats["Balanced Accuracy"] = df_evaluation["Balanced Accuracy"]
    df_stats["F1 Score"] = df_evaluation["F1 Score"]
    
    return df_stats
  
def get_results(aggregate_by_time=True):
    df_times, df_training_stats, df_inference_stats, df_load_stats, df_evaluation = load_files()
        
    if aggregate_by_time:
        print("Aggregating stats by time...")
        
        df_training_stats = aggregate_stats_by_time(df_training_stats)
        df_inference_stats = aggregate_stats_by_time(df_inference_stats)
        df_load_stats = aggregate_stats_by_time(df_load_stats)
    
    print("Sorting dataframes by experiment...")
    df_times = sort_by_experiment(df_times)
    df_training_stats = sort_by_experiment(df_training_stats)
    df_inference_stats = sort_by_experiment(df_inference_stats)
    df_load_stats = sort_by_experiment(df_load_stats)
    df_evaluation = sort_by_experiment(df_evaluation)
        
    # calculate energy consumption in Joules
    print("Calculating energy consumption...")
    df_training_stats = calculate_energy_consumption(df_training_stats, "training", df_times)
    df_inference_stats = calculate_energy_consumption(df_inference_stats, "inference", df_times)
    df_load_stats = calculate_energy_consumption(df_load_stats, "load", df_times)
    
    # put columns of the same statistic together
    df_training_stats, df_inference_stats, df_load_stats = sort_columns(df_training_stats, df_inference_stats, df_load_stats)
    
    # move column "model_size" of df_times to df_load_stats
    df_times, df_load_stats = move_model_size(df_times, df_load_stats)

    # pretty print column names
    df_times, df_training_stats, df_inference_stats, df_load_stats, df_evaluation = pretty_format_column_names(df_times, df_training_stats, df_inference_stats, df_load_stats, df_evaluation)
    
    # reorder columns
    # df_training_stats, df_inference_stats, df_load_stats = reorder_columns(df_training_stats, df_inference_stats, df_load_stats)
    
    # round results
    df_times, df_training_stats, df_inference_stats, df_load_stats, df_evaluation = round_results(df_times, df_training_stats, df_inference_stats, df_load_stats, df_evaluation, decimal_places=3)
    
    # remove unnecesary stats
    df_training_stats, df_inference_stats, df_load_stats = remove_unnecesary_stats(df_training_stats, df_inference_stats, df_load_stats)
    
    # group inference and evaluation
    # df_inference_stats = group_inference_and_evaluation(df_inference_stats, df_evaluation)
    
    # remove Device column
    # df_times = df_times.drop(columns=["Device"])
    # df_training_stats = df_training_stats.drop(columns=["Device"])
    # df_inference_stats = df_inference_stats.drop(columns=["Device"])
    # df_load_stats = df_load_stats.drop(columns=["Device"])
    # df_evaluation = df_evaluation.drop(columns=["Device"])
    
    # remove "EXP" from experiment names
    # df_times["Experiment"] = df_times["Experiment"].str.replace("EXP", "")
    # df_training_stats["Experiment"] = df_training_stats["Experiment"].str.replace("EXP", "")
    # df_inference_stats["Experiment"] = df_inference_stats["Experiment"].str.replace("EXP", "")
    # df_load_stats["Experiment"] = df_load_stats["Experiment"].str.replace("EXP", "")
    # df_evaluation["Experiment"] = df_evaluation["Experiment"].str.replace("EXP", "")
    
    return df_times, df_training_stats, df_inference_stats, df_load_stats, df_evaluation

def print_results(df_times, df_training_stats, df_inference_stats, df_load_stats, df_evaluation):
    print("Results of the benchmark:")
    
    print("Training/Inference/Load times:")
    print(tabulate(df_times))
    
    print("Training stats:")
    print(tabulate(df_training_stats))
    
    print("Inference stats:")
    print(tabulate(df_inference_stats))
    
    print("Load stats:")
    print(tabulate(df_load_stats))
    
    print("Evaluation:")
    print(tabulate(df_evaluation))

def export_results(df_times, df_training_stats, df_inference_stats, df_load_stats, df_evaluation):
    print("Exporting results...")
    
    Path(f"results/{batch_size}").mkdir(parents=True, exist_ok=True)
        
    df_times.to_csv(f"results/{batch_size}/times_{batch_size}.csv", index=False)
    df_training_stats.to_csv(f"results/{batch_size}/training_stats_{batch_size}.csv")
    df_inference_stats.to_csv(f"results/{batch_size}/inference_stats_{batch_size}.csv")
    df_load_stats.to_csv(f"results/{batch_size}/load_stats_{batch_size}.csv")
    df_evaluation.to_csv(f"results/{batch_size}/evaluation_{batch_size}.csv")
    
    print("Results exported to results/ folder.")

def load_processed_results(batch_size):
    df_times = pd.read_csv(f"results/{batch_size}/times_{batch_size}.csv")
    df_training_stats = pd.read_csv(f"results/{batch_size}/training_stats_{batch_size}.csv")
    df_inference_stats = pd.read_csv(f"results/{batch_size}/inference_stats_{batch_size}.csv")
    df_load_stats = pd.read_csv(f"results/{batch_size}/load_stats_{batch_size}.csv")
    df_evaluation = pd.read_csv(f"results/{batch_size}/evaluation_{batch_size}.csv")
    
    return df_times, df_training_stats, df_inference_stats, df_load_stats, df_evaluation

def reorder_stats_columns(df_training_stats, df_inference_stats, df_load_stats):
    # reorder columns
    order = ["Experiment", "Device", "Total Avg. CPU Energy Consumption (J)", "Percentage of Total Avg. CPU Energy Consumption Reduction (%)", "Avg. CPU Power Draw (W)", "Std. Dev. CPU Power Draw (W)", "Max. CPU Power Draw (W)", "Avg. CPU Usage (%)", "Std. Dev. CPU Usage (%)", "Max. CPU Usage (%)", "Avg. CPU Frequency (MHz)", "Std. Dev. CPU Frequency (MHz)", "Max. CPU Frequency (MHz)", "Avg. RAM Memory USS (B)", "Std. Dev. RAM Memory USS (B)", "Max. RAM Memory USS (B)"]
    
    df_training_stats = df_training_stats[order]
    df_inference_stats = df_inference_stats[order]
    df_load_stats = df_load_stats[order + ["Model Size (B)"]]
    
    return df_training_stats, df_inference_stats, df_load_stats

def reorder_times_columns(df_times):
    # reorder columns
    order = ["Experiment", "Device", "Avg. Inference Time (s)", "Std. Dev. Inference Time (s)", "Max. Inference Time (s)", "Avg. Training Time (s)", "Std. Dev. Training Time (s)", "Max. Training Time (s)", "Avg. Load Time (s)", "Std. Dev. Load Time (s)", "Max. Load Time (s)"]
    
    df_times = df_times[order]
    
    return df_times

def prepare_results(batch_sizes):
    _df_times, _df_training_stats, _df_inference_stats, _df_load_stats, df_evaluation_32 = load_processed_results(32)

    for batch_size in batch_sizes:
        df_times, df_training_stats, df_inference_stats, df_load_stats, df_evaluation = load_processed_results(batch_size)
        
        # replace last two column names
        new_column_names = ["Total Avg. CPU Energy Consumption (J)", "Percentage of Total Avg. CPU Energy Consumption Reduction (%)"]

        df_training_stats.columns = df_training_stats.columns[:-2].tolist() + new_column_names
        df_inference_stats.columns = df_inference_stats.columns[:-2].tolist() + new_column_names
        df_load_stats.columns = df_load_stats.columns[:-3].tolist() + new_column_names + ["Model Size (B)"]
        
        # reorder columns
        df_times = reorder_times_columns(df_times)
        df_training_stats, df_inference_stats, df_load_stats = reorder_stats_columns(df_training_stats, df_inference_stats, df_load_stats)

        # round results
        df_times, df_training_stats, df_inference_stats, df_load_stats, df_evaluation = round_results(df_times, df_training_stats, df_inference_stats, df_load_stats, df_evaluation, decimal_places=3)

        # remove unnecesary stats
        df_training_stats, df_inference_stats, df_load_stats = remove_unnecesary_stats(df_training_stats, df_inference_stats, df_load_stats)

        # group inference and evaluation
        df_times = group_with_evaluation(df_times, df_evaluation_32)
        df_training_stats = group_with_evaluation(df_training_stats, df_evaluation_32)
        df_inference_stats = group_with_evaluation(df_inference_stats, df_evaluation_32)
        df_load_stats = group_with_evaluation(df_load_stats, df_evaluation_32)

        # remove Device column
        df_times = df_times.drop(columns=["Device"])
        df_training_stats = df_training_stats.drop(columns=["Device"])
        df_inference_stats = df_inference_stats.drop(columns=["Device"])
        df_load_stats = df_load_stats.drop(columns=["Device"])
        df_evaluation = df_evaluation.drop(columns=["Device"])

        # remove "EXP" from experiment names
        df_times["Experiment"] = df_times["Experiment"].str.replace("EXP", "")
        df_training_stats["Experiment"] = df_training_stats["Experiment"].str.replace("EXP", "")
        df_inference_stats["Experiment"] = df_inference_stats["Experiment"].str.replace("EXP", "")
        df_load_stats["Experiment"] = df_load_stats["Experiment"].str.replace("EXP", "")
        df_evaluation["Experiment"] = df_evaluation["Experiment"].str.replace("EXP", "")

        print("Exporting results...")
        
        revision = "rev6"
        Path(f"results/{revision}/{batch_size}").mkdir(parents=True, exist_ok=True)
        
        # remove unnamed columns
        df_times = df_times.loc[:, ~df_times.columns.str.contains('^Unnamed')]
        df_training_stats = df_training_stats.loc[:, ~df_training_stats.columns.str.contains('^Unnamed')]
        df_inference_stats = df_inference_stats.loc[:, ~df_inference_stats.columns.str.contains('^Unnamed')]
        df_load_stats = df_load_stats.loc[:, ~df_load_stats.columns.str.contains('^Unnamed')]
        df_evaluation = df_evaluation.loc[:, ~df_evaluation.columns.str.contains('^Unnamed')]
        
        # replace "Experiment" by "Opt. Strategy Id."
        df_times = df_times.rename(columns={"Experiment": "Opt. Strategy Id."})
        df_training_stats = df_training_stats.rename(columns={"Experiment": "Opt. Strategy Id."})
        df_inference_stats = df_inference_stats.rename(columns={"Experiment": "Opt. Strategy Id."})
        df_load_stats = df_load_stats.rename(columns={"Experiment": "Opt. Strategy Id."})
        df_evaluation = df_evaluation.rename(columns={"Experiment": "Opt. Strategy Id."})
        
        df_times.to_csv(f"results/{revision}/{batch_size}/times_{batch_size}_final.csv", index=False)
        df_training_stats.to_csv(f"results/{revision}/{batch_size}/training_stats_{batch_size}_final.csv", index=False)
        df_inference_stats.to_csv(f"results/{revision}/{batch_size}/inference_stats_{batch_size}_final.csv", index=False)
        df_load_stats.to_csv(f"results/{revision}/{batch_size}/load_stats_{batch_size}_final.csv", index=False)
        df_evaluation.to_csv(f"results/{revision}/{batch_size}/evaluation_{batch_size}_final.csv", index=False)
        
        print("Results exported to results/ folder.")