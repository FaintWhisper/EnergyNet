def benchmark_training(device, post_training_optimizations=None, training_aware_optimizations=None):
    gpu_memory_usage = get_gpu_memory_system()
    gpu_memory_usage = gpu_memory_usage.replace('MiB', '')
    gpu_memory_usage = float(gpu_memory_usage)

    assert gpu_memory_usage == idle_gpu_memory_usage

    print(f"GPU memory usage: {gpu_memory_usage} MiB")
    
    pid = os.getpid()

    with StatsCollectionManager(test="training", sampling_rate=stats_sampling_rate, pid=pid) as training_scm:
        with measure_time() as training_time_measure:
            # Training
            if training_aware_optimizations is None:
                logging.info("Training without training-aware optimizations")

                with tf.device(device):
                    model = baseline_model(input_dim=len(gf), n_output=2)
                    model, history = train_model(model)
                    tflite_model = export_keras_model_to_tflite(model)
                    model_path = save_model_to_tflite(tflite_model, model_filename="model")
                    # model_path = apply_post_training_optimizations(model, post_training_optimizations)
                    
                    if post_training_optimizations is not None:
                        if "full_integer_quantization" in post_training_optimizations:
                            model_path = full_integer_quantization(model)
                        elif "float16_quantization" in post_training_optimizations:
                            model_path = float16_quantization(model)
                        elif "float16_int8_quantization" in post_training_optimizations:
                            model_path = float16_int8_quantization(model)
                        
                        # print("Exporting TFLite model to ONNX")
                        # onnx_model = export_tflite_model_to_onnx(model_path)
            else:
                with tf.device(device):
                    logging.info("Training with training-aware optimizations")
                    model = baseline_model(input_dim=len(gf), n_output=2)
                    model, history = train_model(model)
                    # model, history = apply_training_aware_optimizations(model, training_aware_optimizations)
                                        
                    if training_aware_optimizations is not None:
                        if "knowledge_distillation" in training_aware_optimizations:
                            if "pruning" in training_aware_optimizations or "quantization_aware_training" in training_aware_optimizations:
                                model_path, _history = knowledge_distillation(model, convert_to_tflite=False)
                            else:
                                model_path, _history = knowledge_distillation(model)
                        
                        if "pruning" in training_aware_optimizations:
                            if "quantization_aware_training" in training_aware_optimizations:
                                model_path, _history = pruning(model, convert_to_tflite=False)
                            else:
                                model_path, _history = pruning(model)
                            
                        if "quantization_aware_training" in training_aware_optimizations:
                            model_path, _history = quantization_aware_training(model)
                        
                        # print("Exporting TFLite model to ONNX")
                        # onnx_model = export_tflite_model_to_onnx(model_path)
    
    gpu_memory_usage = get_gpu_memory_system()
    gpu_memory_usage = gpu_memory_usage.replace('MiB', '')
    gpu_memory_usage = float(gpu_memory_usage)

    print(f"GPU memory usage: {gpu_memory_usage} MiB")

    assert gpu_memory_usage == idle_gpu_memory_usage

    # # Get total parameters count
    # if isinstance(model, tf.keras.Model):
    #     num_params = model.count_params()
    # elif isinstance(model, tf.lite.TFLiteConverter): # NOT SUPPORTED
    #     num_params = model._num_parameters
    # else:
    #     raise Exception(f"Unknown model type: {type(model)}")
    
    # logger.info(f"Number of total parameters: {num_params}")

    # # Save model to ONNX
    # if isinstance(model, tf.keras.Model):
    #     # tflite_model = export_keras_model_to_tflite(model)
    #     # onnx_model = export_keras_model_to_onnx(model)
    #     print("Exporting Keras model to ONNX")
    #     onnx_model = export_keras_model_to_onnx(model)
    # else:
    #     print("Exporting TFLite model to ONNX")
    #     onnx_model = export_tflite_model_to_onnx(model_path)
    
    # save_model(onnx_model, ext="onnx")
    
    gzip_model(model_path)

    # Get training time
    training_time = training_time_measure()

    # Write training time to file
    with open(f"poc_energy_efficiency_crypto/crypto_spider_5g_fcnn_optimized_benchmark_training_time.pkl", "wb") as f:
        pickle.dump({"training_time": training_time}, f)
        

def benchmark_inference(device):
    gpu_memory_usage = get_gpu_memory_system()
    gpu_memory_usage = gpu_memory_usage.replace('MiB', '')
    gpu_memory_usage = float(gpu_memory_usage)

    assert gpu_memory_usage == idle_gpu_memory_usage

    print(f"GPU memory usage: {gpu_memory_usage} MiB")
    
    pid = os.getpid()

    # load model
    # model = load_gzipped_model_from_h5()
    # model = load_model(ext="onnx")
    interpreter = load_model(ext="tflite")
    
    # # resize input
    # batch_size = 256
    # tensor_index = interpreter.get_input_details()[0]["index"]
    # interpreter.resize_tensor_input(tensor_index, [batch_size, 16])
    # interpreter.allocate_tensors()

    # Inference test
    with StatsCollectionManager(test="inference", sampling_rate=stats_sampling_rate, pid=pid) as inference_scm:
        with measure_time() as inference_time_measure:
            with tf.device(device):
                # if post_training_optimizations is None and training_aware_optimizations is None:
                #     perform_inference(model)
                # elif "quantization" in post_training_optimizations or "quantization" in training_aware_optimizations:
                #     perform_inference_quantized(model)
                # else:
                #     perform_inference(model)
                # perform_inference_onnx(model)
                perform_inference_tflite(interpreter, batch_size=256)

    gpu_memory_usage = get_gpu_memory_system()
    gpu_memory_usage = gpu_memory_usage.replace('MiB', '')
    gpu_memory_usage = float(gpu_memory_usage)

    print(f"GPU memory usage: {gpu_memory_usage} MiB")

    assert gpu_memory_usage == idle_gpu_memory_usage

    # Get inference and load times
    inference_time = inference_time_measure()

    # Write inference time to file
    with open(f"poc_energy_efficiency_crypto/crypto_spider_5g_fcnn_optimized_benchmark_inference_time.pkl", "wb") as f:
        pickle.dump({"inference_time": inference_time}, f)

def benchmark_load(device):
    gpu_memory_usage = get_gpu_memory_system()
    gpu_memory_usage = gpu_memory_usage.replace('MiB', '')
    gpu_memory_usage = float(gpu_memory_usage)

    assert gpu_memory_usage == idle_gpu_memory_usage

    print(f"GPU memory usage: {gpu_memory_usage} MiB")
    
    pid = os.getpid()

    # load test
    with StatsCollectionManager(test="load", sampling_rate=stats_sampling_rate, pid=pid) as load_scm:
        with measure_time() as load_time_measure:
            with tf.device(device):
                # if post_training_optimizations is None:
                #     load_gzipped_model()
                # elif "pruning" in post_training_optimizations:
                #     load_gzipped_pruned_model()
                # else:
                #     load_gzipped_model()
                load_model(ext="tflite")
    
    gpu_memory_usage = get_gpu_memory_system()
    gpu_memory_usage = gpu_memory_usage.replace('MiB', '')
    gpu_memory_usage = float(gpu_memory_usage)

    print(f"GPU memory usage: {gpu_memory_usage} MiB")

    assert gpu_memory_usage == idle_gpu_memory_usage

    # Get load time
    load_time = load_time_measure()

    # Write load time to file
    with open(f"poc_energy_efficiency_crypto/crypto_spider_5g_fcnn_optimized_benchmark_load_time.pkl", "wb") as f:
        pickle.dump({"load_time": load_time}, f)

def benchmark(experiment, device="GPU"):
    assert device in ["CPU", "GPU"]

    logger.info("Benchmarking {} on {}".format(experiment, device))

    post_training_optimizations = experiments[experiment]["post_training_optimizations"]
    training_aware_optimizations = experiments[experiment]["training_aware_optimizations"]

    training_times_list = []
    inference_times_list = []
    load_times_list = []
    
    training_stats_list = []
    inference_stats_list = []
    load_stats_list = []
    
    for i in range(0, num_trials):
        logger.info(f"Trial {i}")

        p_training = multiprocessing.Process(target=benchmark_training, args=(device, post_training_optimizations, training_aware_optimizations))
        p_training.start()
        p_training.join()

        p_inference = multiprocessing.Process(target=benchmark_inference, args=(device,))
        p_inference.start()
        p_inference.join()

        p_load = multiprocessing.Process(target=benchmark_load, args=(device,))
        p_load.start()
        p_load.join()
        
        # Read training, inference and load times to file
        with open(f"poc_energy_efficiency_crypto/crypto_spider_5g_fcnn_optimized_benchmark_training_time.pkl", "rb") as f:
            training_time = pickle.load(f)
        
        with open(f"poc_energy_efficiency_crypto/crypto_spider_5g_fcnn_optimized_benchmark_inference_time.pkl", "rb") as f:
            inference_time = pickle.load(f)
        
        with open(f"poc_energy_efficiency_crypto/crypto_spider_5g_fcnn_optimized_benchmark_load_time.pkl", "rb") as f:
            load_time = pickle.load(f)

        inference_time = inference_time["inference_time"]
        training_time = training_time["training_time"]
        load_time = load_time["load_time"]

        # Delete trial times files
        os.remove("poc_energy_efficiency_crypto/crypto_spider_5g_fcnn_optimized_benchmark_training_time.pkl")
        os.remove("poc_energy_efficiency_crypto/crypto_spider_5g_fcnn_optimized_benchmark_inference_time.pkl")
        os.remove("poc_energy_efficiency_crypto/crypto_spider_5g_fcnn_optimized_benchmark_load_time.pkl")
        
        # Read training, inference and load stats to file
        with open('poc_energy_efficiency_crypto/crypto_spider_5g_fcnn_optimized_benchmark_training_stats.pkl', 'rb') as f:
            training_stats = pickle.load(f)
            
        with open('poc_energy_efficiency_crypto/crypto_spider_5g_fcnn_optimized_benchmark_inference_stats.pkl', 'rb') as f:
            inference_stats = pickle.load(f)

        with open('poc_energy_efficiency_crypto/crypto_spider_5g_fcnn_optimized_benchmark_load_stats.pkl', 'rb') as f:
            load_stats = pickle.load(f)
    
        # Delete trial stats files
        os.remove("poc_energy_efficiency_crypto/crypto_spider_5g_fcnn_optimized_benchmark_training_stats.pkl")
        os.remove("poc_energy_efficiency_crypto/crypto_spider_5g_fcnn_optimized_benchmark_inference_stats.pkl")
        os.remove("poc_energy_efficiency_crypto/crypto_spider_5g_fcnn_optimized_benchmark_load_stats.pkl")

        logger.info("Trial {} - Training time: {:.2f}s".format(i, training_time))
        logger.info("Trial {} - Inference time: {:.2f}s".format(i, inference_time))
        logger.info("Trial {} - Load time: {:.2f}s".format(i, load_time))
        logger.info("Trial {} - Total time: {:.2f}s".format(i, training_time + inference_time + load_time))

        # Save training, inference and load times
        training_times_list.append(training_time)
        inference_times_list.append(inference_time)
        load_times_list.append(load_time)

        # Save training, inference and load stats
        training_stats_list.append(training_stats)
        inference_stats_list.append(inference_stats)
        load_stats_list.append(load_stats)
        
        if i != num_trials - 1:
            # Delete models
            delete_model("onnx")
            delete_model("tflite")
            delete_model("h5")

    average_training_time = np.mean(training_times_list)
    std_dev_training_time = np.std(training_times_list)
    max_training_time = np.max(training_times_list)

    average_inference_time = np.mean(inference_times_list)
    std_dev_inference_time = np.std(inference_times_list)
    max_inference_time = np.max(inference_times_list)

    average_load_time = np.mean(load_times_list)
    std_dev_load_time = np.std(load_times_list)
    max_load_time = np.max(load_times_list)

    # Time spent on training
    logger.info("Average training time: {}".format(np.round(average_training_time, 2)))
    logger.info("Standard deviation of training time: {}".format(np.round(std_dev_training_time, 2)))
    logger.info("Max. training time: {}".format(np.round(max_training_time, 2)))

    # Time spent on inference
    logger.info("Average inference time: {}".format(np.round(average_inference_time, 2)))
    logger.info("Standard deviation of inference time: {}".format(np.round(std_dev_inference_time, 2)))
    logger.info("Max. inference time: {}".format(np.round(max_inference_time, 2)))

    # Time spent on loading
    logger.info("Average load time: {}".format(np.round(average_load_time, 2)))
    logger.info("Standard deviation of load time: {}".format(np.round(std_dev_load_time, 2)))
    logger.info("Max. load time: {}".format(np.round(max_load_time, 2)))

    # Get average training, inference and load times
    average_training_stats_list = get_average_stats(training_stats_list, average_training_time)
    std_dev_training_stats_list = get_std_dev_stats(training_stats_list, std_dev_training_time)
    max_training_stats_list = get_max_stats(training_stats_list, max_training_time)

    average_inference_stats_list = get_average_stats(inference_stats_list, average_inference_time)
    std_dev_inference_stats_list = get_std_dev_stats(inference_stats_list, std_dev_inference_time)
    max_inference_stats_list = get_max_stats(inference_stats_list, max_inference_time)

    average_load_stats_list = get_average_stats(load_stats_list, average_load_time)
    std_dev_load_stats_list = get_std_dev_stats(load_stats_list, std_dev_load_time)
    max_load_stats_list = get_max_stats(load_stats_list, max_load_time)

    # Get model size
    model_size = get_gzipped_model_size(ext="tflite")
    logger.info(f"Model size (gzip): {model_size}")

    # Save training, inference and load metrics to dataframe
    stats = training_stats_list[0][0].keys()
    
    average_traning_stats_names = [f"average_training_{stat}" for stat in stats]
    std_dev_training_stats_names = [f"std_dev_training_{stat}" for stat in stats]
    max_training_stats_names = [f"max_training_{stat}" for stat in stats]
    
    average_inference_stats_names = [f"average_inference_{stat}" for stat in stats]
    std_dev_inference_stats_names = [f"std_dev_inference_{stat}" for stat in stats]
    max_inference_stats_names = [f"max_inference_{stat}" for stat in stats]
    
    average_load_stats_names = [f"average_load_{stat}" for stat in stats]
    std_dev_load_stats_names = [f"std_dev_load_{stat}" for stat in stats]
    max_load_stats_names = [f"max_load_{stat}" for stat in stats]

    df_times_columns = ["experiment", "device", "average_training_time", "std_dev_training_time", "max_training_time", "average_inference_time", "std_dev_inference_time", "max_inference_time", "average_load_time", "std_dev_load_time", "max_load_time", "model_size"]
    df_times = pd.DataFrame(columns=df_times_columns)
    times_row = [experiment, device, average_training_time, std_dev_training_time, max_training_time, average_inference_time, std_dev_inference_time, max_inference_time, average_load_time, std_dev_load_time, max_load_time, model_size]
    df_times.loc[0] = times_row

    df_training_stats_columns = ["experiment", "device", "snapshot", *average_traning_stats_names, *std_dev_training_stats_names, *max_training_stats_names]
    df_training_stats = pd.DataFrame(columns=df_training_stats_columns)
    
    for index, _snapshot in enumerate(average_training_stats_list):
        row = np.array([experiment, device, index])
        row = np.append(row, average_training_stats_list[index])
        row = np.append(row, std_dev_training_stats_list[index])
        row = np.append(row, max_training_stats_list[index])
    
        df_training_stats.loc[index] = row

    df_inference_stats_columns = ["experiment", "device", "snapshot", *average_inference_stats_names, *std_dev_inference_stats_names, *max_inference_stats_names]
    df_inference_stats = pd.DataFrame(columns=df_inference_stats_columns)
    
    for index, _snapshot in enumerate(average_inference_stats_list):
        row = np.array([experiment, device, index])
        row = np.append(row, average_inference_stats_list[index])
        row = np.append(row, std_dev_inference_stats_list[index])
        row = np.append(row, max_inference_stats_list[index])

        df_inference_stats.loc[index] = row

    df_load_stats_columns = ["experiment", "device", "snapshot", *average_load_stats_names, *std_dev_load_stats_names, *max_load_stats_names]
    df_load_stats = pd.DataFrame(columns=df_load_stats_columns)
    
    for index, _snapshot in enumerate(average_load_stats_list):
        row = np.array([experiment, device, index])
        row = np.append(row, average_load_stats_list[index])
        row = np.append(row, std_dev_load_stats_list[index])
        row = np.append(row, max_load_stats_list[index])

        df_load_stats.loc[index] = row

    # # Save average and std_dev of training, inference and load stats to log file
    # save_stats_to_logfile(test="training", average_stats=average_training_stats_list, std_dev_stats=std_dev_training_stats_list)
    # save_stats_to_logfile(test="inference", average_stats=average_inference_stats_list, std_dev_stats=std_dev_inference_stats_list)
    # save_stats_to_logfile(test="load", average_stats=average_load_stats_list, std_dev_stats=std_dev_load_stats_list)

    return df_times, df_training_stats, df_inference_stats, df_load_stats

def run_benchmark(experiments, devices):
    for experiment in experiments:
        for device in devices:
            print(f"Running experiment {experiment} on device {device}")
            df_times, df_training_stats, df_inference_stats, df_load_stats = benchmark(experiment=experiment, device=device)
            df_evaluation = perform_evaluation_tflite(experiment=experiment, device=device)

            # load times dataframe from file and append new df
            if os.path.exists("poc_energy_efficiency_crypto/crypto_spider_5g_fcnn_optimized_benchmark_times.csv"):
                df_times.to_csv("poc_energy_efficiency_crypto/crypto_spider_5g_fcnn_optimized_benchmark_times.csv", mode="a", header=False, index=False)
            else:
                df_times.to_csv("poc_energy_efficiency_crypto/crypto_spider_5g_fcnn_optimized_benchmark_times.csv", index=False)

            # load training, inference and load stats dataframe from file and append new df
            if os.path.exists("poc_energy_efficiency_crypto/crypto_spider_5g_fcnn_optimized_benchmark_training_stats.csv"):
                df_training_stats.to_csv("poc_energy_efficiency_crypto/crypto_spider_5g_fcnn_optimized_benchmark_training_stats.csv", mode="a", header=False, index=False)
            else:
                df_training_stats.to_csv("poc_energy_efficiency_crypto/crypto_spider_5g_fcnn_optimized_benchmark_training_stats.csv", index=False)
            
            if os.path.exists("poc_energy_efficiency_crypto/crypto_spider_5g_fcnn_optimized_benchmark_inference_stats.csv"):
                df_inference_stats.to_csv("poc_energy_efficiency_crypto/crypto_spider_5g_fcnn_optimized_benchmark_inference_stats.csv", mode="a", header=False, index=False)
            else:
                df_inference_stats.to_csv("poc_energy_efficiency_crypto/crypto_spider_5g_fcnn_optimized_benchmark_inference_stats.csv", index=False)
            
            if os.path.exists("poc_energy_efficiency_crypto/crypto_spider_5g_fcnn_optimized_benchmark_load_stats.csv"):
                df_load_stats.to_csv("poc_energy_efficiency_crypto/crypto_spider_5g_fcnn_optimized_benchmark_load_stats.csv", mode="a", header=False, index=False)
            else:
                df_load_stats.to_csv("poc_energy_efficiency_crypto/crypto_spider_5g_fcnn_optimized_benchmark_load_stats.csv", index=False)
            
            # load evaluation dataframe from file and append new df
            if os.path.exists("poc_energy_efficiency_crypto/crypto_spider_5g_fcnn_optimized_benchmark_evaluation.csv"):
                df_evaluation.to_csv("poc_energy_efficiency_crypto/crypto_spider_5g_fcnn_optimized_benchmark_evaluation.csv", mode="a", header=False, index=False)
            else:
                df_evaluation.to_csv("poc_energy_efficiency_crypto/crypto_spider_5g_fcnn_optimized_benchmark_evaluation.csv", index=False)
                
            
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
    display(df_times)
    
    print("Training stats:")
    display(df_training_stats)
    
    print("Inference stats:")
    display(df_inference_stats)
    
    print("Load stats:")
    display(df_load_stats)
    
    print("Evaluation:")
    display(df_evaluation)

def export_results(df_times, df_training_stats, df_inference_stats, df_load_stats, df_evaluation):
    print("Exporting results...")
    
    Path(f"results/{batch_size}").mkdir(parents=True, exist_ok=True)
        
    df_times.to_csv(f"results/{batch_size}/times_{batch_size}.csv", index=False)
    df_training_stats.to_csv(f"results/{batch_size}/training_stats_{batch_size}.csv")
    df_inference_stats.to_csv(f"results/{batch_size}/inference_stats_{batch_size}.csv")
    df_load_stats.to_csv(f"results/{batch_size}/load_stats_{batch_size}.csv")
    df_evaluation.to_csv(f"results/{batch_size}/evaluation_{batch_size}.csv")
    
    print("Results exported to results/ folder.")