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
                