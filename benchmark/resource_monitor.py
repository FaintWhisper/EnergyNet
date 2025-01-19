import psutil
import numpy as np
import socket
import subprocess
import time
import multiprocessing
import os
import re
from pathlib import Path
import pandas as pd
import pickle
import copy

# Socket for sending data from powerstat process to main process
HOST = "127.0.0.1"

# Define the measurement tool that will be used to gather power consumption data
power_consumption_measurement_tool = "powerstat"

# RAM
def get_ram_memory_uss(pid):
    process = psutil.Process(pid)
    
    return str(process.memory_full_info().uss / (1024*1024)) + ' MB'

def get_ram_memory_rss(pid):
    process = psutil.Process(pid)
    
    return str(process.memory_full_info().rss / (1024*1024)) + ' MB'

def get_ram_memory_vms(pid):
    process = psutil.Process(pid)
    
    return str(process.memory_full_info().vms / (1024*1024)) + ' MB'

def get_ram_memory_pss(pid):
    process = psutil.Process(pid)
    
    return str(process.memory_full_info().pss / (1024*1024)) + ' MB'

# CPU
def get_cpu_usage(pid):
    process = psutil.Process(pid)
    
    return str(process.cpu_percent(interval=0.5) / psutil.cpu_count()) + ' %'

def get_cpu_freq():    
    return str(psutil.cpu_freq()[0]) + " MHz"

def perf(PORT):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((HOST, PORT))
    cmd = "echo pirata.lab | sudo -S -p \"\" perf stat -e power/energy-cores/,power/energy-pkg/"
    out = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE)
    s.sendall(out.stdout)
    
    s.close()
    
def kill_perf():
    cmd = "echo pirata.lab | sudo -S -p \"\" pkill perf"
    subprocess.run(cmd, shell=True)

def powerstat(PORT):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((HOST, PORT))
    cmd = "echo pirata.lab | sudo -S -p \"\" powerstat -R 0.5 120"
    out = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE)
    s.sendall(out.stdout)
    
    s.close()
    
def kill_powerstat():
    cmd = "echo pirata.lab | sudo -S -p \"\" pkill powerstat"
    subprocess.run(cmd, shell=True)
    
def get_cpu_power_draw():    
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    PORT = np.random.randint(10000, 20000)
    s.bind((HOST, PORT))
    s.listen()
    
    if power_consumption_measurement_tool == "powerstat":
        p = multiprocessing.Process(target=powerstat, args=(PORT,))
    elif power_consumption_measurement_tool == "perf":
        p = multiprocessing.Process(target=perf, args=(PORT,))
        
    p.start()
    conn, _addr = s.accept()
    time.sleep(max(1, stats_sampling_rate / 2))
    
    if power_consumption_measurement_tool == "powerstat":
        q = multiprocessing.Process(target=kill_powerstat)
    elif power_consumption_measurement_tool == "perf":
        q = multiprocessing.Process(target=kill_perf)
        
    q.start()
    
    out = conn.recv(2048).decode()
    
    power_consumption = re.findall(r'CPU: (.+?) Watts', out)[0].strip() + " W"
    
    s.close()
    p.terminate()
    q.terminate()
    
    return power_consumption

# IO
def get_io_usage(pid):
    process = psutil.Process(pid)
    
    io_counters = process.io_counters()
    io_usage_process = io_counters[2] + io_counters[3] # read_bytes + write_bytes
    disk_io_counter = psutil.disk_io_counters()
    disk_io_total = disk_io_counter[2] + disk_io_counter[3] # read_bytes + write_bytes
    io_usage_process = io_usage_process / disk_io_total * 100
    io_usage_process = np.round(io_usage_process, 2)
    io_usage_process = str(io_usage_process) + " %"

    return io_usage_process

def get_bytes_written(pid):
    process = psutil.Process(pid)
    
    io_counters = process.io_counters()
    process_bytes_written = io_counters[3]
    total_bytes_written = psutil.disk_io_counters()[3]
    process_bytes_written = process_bytes_written / total_bytes_written * 100
    process_bytes_written = np.round(process_bytes_written, 2)
    process_bytes_written = str(process_bytes_written) + " %"

    return process_bytes_written

def get_bytes_read(pid):
    process = psutil.Process(pid)
    
    io_counters = process.io_counters()
    process_bytes_read = io_counters[2]
    total_bytes_read = psutil.disk_io_counters()[2]
    process_bytes_read = process_bytes_read / total_bytes_read * 100
    process_bytes_read = np.round(process_bytes_read, 2)
    process_bytes_read = str(process_bytes_read) + " %"

    return process_bytes_read

# GPU
get_gpu_memory_system = lambda: os.popen('nvidia-smi --query-gpu=memory.used --format=csv,noheader').read().split('\n')[0] # SYSTEM WIDE
def get_gpu_memory(pid):
    output = os.popen('nvidia-smi | awk \'/' + str(pid) + '/{print $8}\'').read().split('\n')[0]
    output = "0 MiB" if output == "" else output.replace("MiB", "") + " MiB"

    return output

get_gpu_usage = lambda: os.popen('nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader').read().split('\n')[0]
get_gpu_freq = lambda: os.popen('nvidia-smi --query-gpu=clocks.gr --format=csv,noheader').read().split('\n')[0]
get_gpu_power_draw = lambda: os.popen('nvidia-smi --query-gpu=power.draw --format=csv,noheader').read().split('\n')[0]

test_pid = os.getpid()

# Test functions
print("RAM Memory Usage (USS): " + get_ram_memory_uss(test_pid))
print("RAM Memory Usage (RSS): " + get_ram_memory_rss(test_pid))
print("RAM Memory Usage (VMS): " + get_ram_memory_vms(test_pid))
print("RAM Memory Usage (PSS): " + get_ram_memory_pss(test_pid))
# print("RAM Power Draw: " + get_cpu_power_draw()[1])
print("CPU Usage: " + get_cpu_usage(test_pid))
print("CPU Frequency: " + get_cpu_freq())
# print("CPU Cores Power Draw: " + get_cpu_power_draw()[0])
# print("CPU Package Power Draw: " + get_cpu_power_draw()[2])
print("CPU Power Usage: " + get_cpu_power_draw())
print("I/O Usage: " + get_io_usage(test_pid))
print("Bytes Written to disk: " + str(get_bytes_written(test_pid)))
print("Bytes Read to disk: " + str(get_bytes_read(test_pid)))
print("GPU Memory Usage: " + get_gpu_memory(test_pid))
print("GPU Usage: " + get_gpu_usage())
print("GPU Frequency: " + get_gpu_freq())
print("GPU Power Draw: " + get_gpu_power_draw())

def get_stats(pid):
    stats = {}
    stats['ram_memory_uss'] = get_ram_memory_uss(pid)
    stats['ram_memory_rss'] = get_ram_memory_rss(pid)
    stats['ram_memory_vms'] = get_ram_memory_vms(pid)
    stats['ram_memory_pss'] = get_ram_memory_pss(pid)
    # stats["ram_power_draw"] = get_cpu_power_draw()[1]
    stats['cpu_usage'] = get_cpu_usage(pid)
    stats['cpu_freq'] = get_cpu_freq()
    # stats['cpu_cores_power_draw'] = get_cpu_power_draw()[0]
    # stats['cpu_package_power_draw'] = get_cpu_power_draw()[2]
    stats['cpu_power_draw'] = get_cpu_power_draw()
    stats['io_usage'] = get_io_usage(pid)
    stats['bytes_written'] = get_bytes_written(pid)
    stats['bytes_read'] = get_bytes_read(pid)
    stats['gpu_memory'] = get_gpu_memory(pid)
    stats['gpu_usage'] = get_gpu_usage()
    stats['gpu_freq'] = get_gpu_freq()
    stats['gpu_power_draw'] = get_gpu_power_draw()

    return stats

def sample_stats(test, sampling_rate, pid, directory):
    print(f"test: {test}")
    
    stats_list = []
    started = False
    
    Path(directory).mkdir(parents=True, exist_ok=True)

    while True:
        stats = get_stats(pid)
        stats_list.append(stats)

        # write stats to pickle file
        with open(f"{directory}/crypto_spider_5g_fcnn_optimized_benchmark_{test}_stats.pkl", 'wb') as f:
            pickle.dump(stats_list, f)

        if not started:
            # write file started.txt to signal that the sampling has started
            with open(f"started_{test}.txt", 'w') as f:
                f.write("STARTED")
            
            print("\nStats sampling started")

        started = True

        # check if file "stop.txt" exists
        if os.path.isfile(f"stop_{test}.txt"):
            print("Stats sampling stopped")
            os.remove(f"stop_{test}.txt")

            break
        else:
            time.sleep(sampling_rate)

def get_stats_background(test, sampling_rate, pid, directory):
    proc = multiprocessing.Process(target=sample_stats, args=(test, sampling_rate, pid, directory))
    proc.start()

    return proc

def strip_units(x):
    return float(x.split(' ')[0])

def agg_stats(agg_func, stats_list, average_time_spent):
    average_stats = copy.deepcopy(stats_list)

    # strip units of the stats of every trial in stats_list
    for trial in average_stats:
        for snapshot in trial:
            for stat in snapshot:
                stats_value = snapshot[stat]
                stats_value_stripped = strip_units(stats_value)
                snapshot[stat] = stats_value_stripped

    trials_list = []
    
    # convert to a numpy array
    for trial in average_stats:
        df = pd.DataFrame(trial)
        trial = df.to_numpy()
        
        trials_list.append(trial)
    
    trials_list_np = np.array(trials_list)
    
    print("trials_list_np.shape: {}".format(trials_list_np.shape))
    
    # fill first axis of trials_list_np with NaNs until all trials have the same length
    # trials_list_np = np.array([np.pad(trial, ((0, trials_list_np.shape[0] - trial.shape[0]), (0, 0)), 'constant', constant_values=np.nan) for trial in trials_list_np])
    max_length = max([trial.shape[0] for trial in trials_list_np])
    trials_list_np_filled = []
    
    for trial in trials_list_np:
        trial_length = trial.shape[0]
        
        if trial_length < max_length:
            print(f"Trial length ({trial_length}) is smaller than max length ({max_length}). Filling with NaNs...")
        
            # fill first axis of trial with NaNs until trial has the same length as the longest trial
            trial = np.pad(trial, ((0, max_length - trial_length), (0, 0)), 'constant', constant_values=np.nan)
            
            print("trial.shape: {}".format(trial.shape))
        
        trials_list_np_filled.append(trial)
    
    trials_list_np_filled = np.array(trials_list_np_filled)
    
    print("trials_list_np_filled.shape: {}".format(trials_list_np_filled.shape))

    average_stats_np = agg_func(trials_list_np_filled, axis=0)

    print("average_stats_np.shape: {}".format(average_stats_np.shape))

    return average_stats_np

def get_average_stats(stats_list, average_time_spent):
    return agg_stats(agg_func=np.nanmean, stats_list=stats_list, average_time_spent=average_time_spent)

def get_std_dev_stats(stats_list, average_time_spent):
    return agg_stats(agg_func=np.nanstd, stats_list=stats_list, average_time_spent=average_time_spent)

def get_max_stats(stats_list, average_time_spent):
    return agg_stats(agg_func=np.nanmax, stats_list=stats_list, average_time_spent=average_time_spent)

def save_stats_to_logfile(test, average_stats, std_dev_stats, max_stats):
    units = {'ram_memory_uss': 'MB', 'ram_memory_rss': 'MB', 'ram_memory_vms': 'MB', 'ram_memory_pss': 'MB', 'cpu_usage': '%', 'cpu_freq': 'MHz', 'cpu_power_draw': 'W', 'io_usage': '%', 'bytes_written': 'MB', 'bytes_read': 'MB', 'gpu_memory': 'MB', 'gpu_usage': '%', 'gpu_freq': 'MHz', 'gpu_power_draw': 'W'}

    for key in average_stats.keys():
        logger.info(f'[{test}] {key} (average): {average_stats[key]} {units[key]}')
        logger.info(f'[{test}] {key} (std_dev): {std_dev_stats[key]} {units[key]}')
        logger.info(f'[{test}] {key} (max): {max_stats[key]} {units[key]}')