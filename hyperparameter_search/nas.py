# #### NAS

# In[ ]:


# find the student model structure by searching for the model
num_layers_grid = [1, 2, 3, 4]
num_neurons_grid = [4, 8, 16, 32, 64]

# make all possible combinations of the model structure (num_layers, num_neuron_for_each_layer)
combinations = []

for num_layers in num_layers_grid:
   product = list(itertools.product(num_neurons_grid, repeat=num_layers))
   combinations.append(product)

# build the models
for num_layers in range(0, len(combinations)):
   print(f"num_layers: {num_layers + 1}")
   
   for index, num_neurons in enumerate(combinations[num_layers]):
       print(f"index: {index}")
       neurons = combinations[num_layers][index]
       
       student = Sequential()
       
       for layer, neurons_layer in enumerate(neurons):
           print(f"- layer: {layer}, neurons_layer: {neurons_layer}")
       
       print("")


# In[ ]:


num_trials = 5


# In[ ]:


def test_nas(device="CPU"):
    teacher = baseline_model(input_dim=len(gf), n_output=2)
    teacher, history = train_model(teacher)

    num_trials = 5
    
    logger.info("Starting test_nas")
    
    pid = os.getpid()
    
    model_energy_consumption = {}
    
    Path(f"poc_energy_efficiency_crypto/NAS").mkdir(parents=True, exist_ok=True)
    
    with tf.device(device):
        with StatsCollectionManager(directory="NAS", test="NAS", sampling_rate=stats_sampling_rate, pid=pid) as nas_scm:
            with measure_time() as nas_time_measure:
                logger.info("Applying Knowledge Distillation (with NAS)")
        
                # find the student model structure by searching for the model
                num_layers_grid = [1, 2, 3]
                num_neurons_grid = [4, 8, 16, 32]
                
                # num_layers_grid = [1,]
                # num_neurons_grid = [4,]

                # make all possible combinations of the model structure (num_layers, num_neuron_for_each_layer)
                combinations = []

                for num_layers in num_layers_grid:
                    product = list(itertools.product(num_neurons_grid, repeat=num_layers))
                    combinations.append(product)

                # build the models
                for num_layers in range(0, len(combinations)):
                    logger.info(f"num_layers: {num_layers + 1}")
                    
                    for index, num_neurons in enumerate(combinations[num_layers]):
                        neurons = combinations[num_layers][index]
                        logger.info(f"neurons: {num_neurons}")
                        
                        student = Sequential()
                        student.add(tf.keras.Input(shape=(len(gf),)))
                        
                        for layer, neurons_layer in enumerate(neurons):
                            logger.info(f"layer: {layer}")
                            logger.info(f"neurons_layer: {neurons_layer}")
                            
                            student.add(Dense(neurons_layer, activation='relu'))
                        
                        student.add(Dense(2, activation='softmax'))
                        student.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
                        student.summary()

                        # Initialize and compile distiller
                        distiller = Distiller(student=student, teacher=teacher)
                        distiller.compile(
                            optimizer=keras.optimizers.Adam(),
                            metrics=[keras.metrics.CategoricalAccuracy()],
                            student_loss_fn=keras.losses.CategoricalCrossentropy(from_logits=False),
                            distillation_loss_fn=keras.losses.KLDivergence(),
                            alpha=0.1,
                            temperature=3,
                        )

                        es = EarlyStopping(
                            monitor="val_student_loss",
                            mode="min",
                            patience=es_patience,
                            restore_best_weights=es_restore_best_weights,
                        )
                        _history = distiller.fit(
                            data_transformed,
                            pd.get_dummies(df_train["tag"]),
                            validation_split=validation_split,
                            epochs=epochs,
                            batch_size=batch_size,
                            callbacks=[es],
                            verbose=0,
                        )
                                                
                        # save the model
                        comb_id = f"{num_layers + 1}_{index}"
                        distiller.student.save(f"poc_energy_efficiency_crypto/NAS/models/model_{comb_id}.h5")
                        
                        for i in range(0, num_trials):
                            logger.info(f"Trial Evaluation: {i}")
                        
                            # evaluate the model
                            data_transformed_test = standard.transform(df_test[gf])
                            
                            with StatsCollectionManager(directory="NAS", test=f"evaluation_nas_{comb_id}_{i}", sampling_rate=stats_sampling_rate, pid=pid) as evaluation_nas_scm:
                                with measure_time() as evaluation_nas_time_measure:
                                    predictions = distiller.student.predict(data_transformed_test)
                            
                            evaluation_nas_time = evaluation_nas_time_measure()
                            
                            # Write evaluation nas time to file
                            with open(f"poc_energy_efficiency_crypto/NAS/crypto_spider_5g_fcnn_optimized_benchmark_evaluation_nas_time_{comb_id}_{i}.pkl", "wb") as f:
                                pickle.dump({"evaluation_nas_time": evaluation_nas_time}, f)
                                
                        if ml_task == "binary_classification":
                            predictions = np.argmax(predictions, axis=1)
                            
                            accuracy = accuracy_score(df_test["tag"], predictions)
                            f1 = f1_score(df_test["tag"], predictions, average="weighted")
                            auc = roc_auc_score(df_test["tag"], predictions)
                            recall = recall_score(df_test["tag"], predictions, average="weighted")
                            precision = precision_score(df_test["tag"], predictions, average="weighted")
                            balanced_accuracy = balanced_accuracy_score(df_test["tag"], predictions)
                            matthews = matthews_corrcoef(df_test["tag"], predictions)

                            logger.info(f"Accuracy: {accuracy}")
                            logger.info(f"F1 score: {f1}")
                            logger.info(f"AUC: {auc}")
                            logger.info(f"Recall: {recall}")
                            logger.info(f"Precision: {precision}")
                            logger.info(f"Balanced accuracy: {balanced_accuracy}")
                            logger.info(f"Matthews correlation coefficient: {matthews}")
                            
                            test_results = {
                                "accuracy": accuracy,
                                "f1": f1,
                                "auc": auc,
                                "recall": recall,
                                "precision": precision,
                                "balanced_accuracy": balanced_accuracy,
                                "matthews": matthews
                            }
                        elif ml_task == "regression":
                            mae = mean_absolute_error(df_test["tag"], predictions)
                            mse = mean_squared_error(df_test["tag"], predictions)
                            mape = mean_absolute_percentage_error(df_test["tag"], predictions)
                            smape = 1/len(df_test["tag"]) * np.sum(2 * np.abs(predictions - df_test["tag"]) / (np.abs(predictions) + np.abs(df_test["tag"])))
                            
                            logger.info(f"MAE: {mae}")
                            logger.info(f"MSE: {mse}")
                            logger.info(f"MAPE: {mape}")
                            logger.info(f"SMAPE: {smape}")
                            
                            test_results = {
                                "mae": mae,
                                "mse": mse,
                                "mape": mape,
                                "smape": smape
                            }
                            
                        with open(f"poc_energy_efficiency_crypto/NAS/crypto_spider_5g_fcnn_optimized_benchmark_evaluation_nas_test_results_{comb_id}.pkl", "wb") as f:
                            pickle.dump(test_results, f)

                        # accuracy = accuracy_score(df_test['tag'], np.argmax(preds, axis=1))
                        # balanced_accuracy = balanced_accuracy_score(df_test['tag'], np.argmax(preds, axis=1))
                        # f1 = f1_score(df_test['tag'], np.argmax(preds, axis=1))

                        # print(f"Accuracy: {accuracy}")
                        # print(f"Balanced accuracy: {balanced_accuracy}")
                        # print(f"F1 score: {f1}")
                        
                        # # save the results
                        # test_results = [accuracy, balanced_accuracy, f1]
                        
                        # with open(f"poc_energy_efficiency_crypto/crypto_spider_5g_fcnn_optimized_benchmark_evaluation_nas_test_results_{comb_id}.pkl", "wb") as f:
                        #     pickle.dump({"test_results": test_results}, f)
                                    
                        evaluation_nas_times_list = []
                        evaluation_nas_stats_list = []
                        
                        for i in range(0, num_trials):
                            logger.info(f"Trial Evaluation: {i}. Comb ID: {comb_id}")
                                
                            # Read nas time from file
                            with open(f"poc_energy_efficiency_crypto/NAS/crypto_spider_5g_fcnn_optimized_benchmark_evaluation_nas_time_{comb_id}_{i}.pkl", "rb") as f:
                                evaluation_nas_time = pickle.load(f)

                            evaluation_nas_time = evaluation_nas_time["evaluation_nas_time"]
                            evaluation_nas_time = float(evaluation_nas_time)

                            # # Delete nas time file
                            # os.remove(f"poc_energy_efficiency_crypto/crypto_spider_5g_fcnn_optimized_benchmark_evaluation_nas_time_{comb_id}_{i}.pkl")
                                    
                            # Read nas stats from file
                            with open(f'poc_energy_efficiency_crypto/NAS/crypto_spider_5g_fcnn_optimized_benchmark_evaluation_nas_{comb_id}_{i}_stats.pkl', 'rb') as f:
                                evaluation_nas_stats = pickle.load(f)
                                
                            # # Delete nas stats file
                            # os.remove(f"poc_energy_efficiency_crypto/crypto_spider_5g_fcnn_optimized_benchmark_evaluation_nas_stats_{comb_id}_{i}.pkl")
                            
                            logger.info("Trial {}. Comb ID {}. Evaluation NAS time: {}".format(i, comb_id, evaluation_nas_time))

                            # Save NAS times
                            evaluation_nas_times_list.append(evaluation_nas_time)

                            # Save NAS stats
                            evaluation_nas_stats_list.append(evaluation_nas_stats)

                        average_evaluation_nas_time = np.mean(evaluation_nas_times_list)
                        std_dev_evaluation_nas_time = np.std(evaluation_nas_times_list)
                        max_evaluation_nas_time = np.max(evaluation_nas_times_list)

                        # Time spent on NAS
                        logger.info(f"Comb ID: {comb_id}. " + "Average NAS time: {}".format(np.round(average_evaluation_nas_time, 2)))
                        logger.info(f"Comb ID: {comb_id}. " + "Standard deviation of NAS time: {}".format(np.round(std_dev_evaluation_nas_time, 2)))
                        logger.info(f"Comb ID: {comb_id}. " + "Max. NAS time: {}".format(np.round(max_evaluation_nas_time, 2)))

                        # Get average NAS metrics
                        average_evaluation_nas_stats_list = get_average_stats(evaluation_nas_stats_list, average_evaluation_nas_time)
                        std_dev_evaluation_nas_stats_list = get_std_dev_stats(evaluation_nas_stats_list, std_dev_evaluation_nas_time)
                        max_evaluation_nas_stats_list = get_max_stats(evaluation_nas_stats_list, max_evaluation_nas_time)
                        
                        # Save NAS metrics to dataframe
                        stats = evaluation_nas_stats_list[0][0].keys()

                        averag_evaluation_nas_stats_names = [f"average_evaluation_nas_{stat}" for stat in stats]
                        std_dev_evaluation_nas_stats_names = [f"std_dev_evaluation_nas_{stat}" for stat in stats]
                        max_evaluation_nas_stats_names = [f"max_evaluation_nas_{stat}" for stat in stats]

                        df_evaluation_nas_times_columns = ["experiment", "device", "average_evaluation_nas_time", "std_dev_evaluation_nas_time", "max_evaluation_nas_time"]
                        df_evaluation_nas_times = pd.DataFrame(columns=df_evaluation_nas_times_columns)
                        
                        experiment = f"evaluation_nas_{comb_id}"
                        evaluation_nas_times_row = [experiment, device, average_evaluation_nas_time, std_dev_evaluation_nas_time, max_evaluation_nas_time]
                        df_evaluation_nas_times.loc[0] = evaluation_nas_times_row

                        df_evaluation_nas_stats_columns = ["experiment", "device", "snapshot", *averag_evaluation_nas_stats_names, *std_dev_evaluation_nas_stats_names, *max_evaluation_nas_stats_names]
                        df_evaluation_nas_stats = pd.DataFrame(columns=df_evaluation_nas_stats_columns)
                        
                        for index, _snapshot in enumerate(average_evaluation_nas_stats_list):
                            row = np.array([experiment, device, index])
                            row = np.append(row, average_evaluation_nas_stats_list[index])
                            row = np.append(row, std_dev_evaluation_nas_stats_list[index])
                            row = np.append(row, max_evaluation_nas_stats_list[index])

                            df_evaluation_nas_stats.loc[index] = row
                            
                        assert df_evaluation_nas_times.shape[0] == 1
                        assert df_evaluation_nas_stats.shape[0] == len(average_evaluation_nas_stats_list)
                        assert df_evaluation_nas_stats.shape[1] == len(df_evaluation_nas_stats_columns)
                        
                        display(df_evaluation_nas_times)
                        display(df_evaluation_nas_stats)

                        # calculate the average of the df_evaluation_nas_stats average_evaluation_nas_cpu_power_draw and average_evaluation_nas_gpu_power_draw columns
                        global_average_average_evaluation_nas_cpu_power_draw = df_evaluation_nas_stats[f"average_evaluation_nas_cpu_power_draw"].astype(float).mean()
                        global_average_average_evaluation_nas_gpu_power_draw = df_evaluation_nas_stats[f"average_evaluation_nas_gpu_power_draw"].astype(float).mean()
                        
                        # check that both columns have a single scalar value
                        assert np.isscalar(global_average_average_evaluation_nas_cpu_power_draw)
                        assert np.isscalar(global_average_average_evaluation_nas_gpu_power_draw)
                        
                        average_evaluation_nas_exp_duration = df_evaluation_nas_times[f"average_evaluation_nas_time"].values[0]
                        
                        print("Average evaluation NAS experiment duration: {}".format(average_evaluation_nas_exp_duration))
                        print("Average evaluation NAS CPU power draw: {}".format(global_average_average_evaluation_nas_cpu_power_draw))
                        print("Average evaluation NAS GPU power draw: {}".format(global_average_average_evaluation_nas_gpu_power_draw))
                        
                        total_average_cpu_energy_consumption = global_average_average_evaluation_nas_cpu_power_draw * average_evaluation_nas_exp_duration
                        total_average_gpu_energy_consumption = global_average_average_evaluation_nas_gpu_power_draw * average_evaluation_nas_exp_duration
                        
                        model_energy_consumption[comb_id] = {
                            "cpu": total_average_cpu_energy_consumption,
                            "gpu": total_average_gpu_energy_consumption
                        }
                        
                        print("Model energy consumption:\n{}".format(model_energy_consumption))
                    
                # normalize energy consumption between 0 and 1
                normalized_model_energy_consumption = deepcopy(model_energy_consumption)
                
                max_cpu_energy_consumption = max([model_energy_consumption[comb_id]["cpu"] for comb_id in model_energy_consumption])
                max_gpu_energy_consumption = max([model_energy_consumption[comb_id]["gpu"] for comb_id in model_energy_consumption])
                
                for comb_id in model_energy_consumption:
                    normalized_model_energy_consumption[comb_id]["cpu"] = model_energy_consumption[comb_id]["cpu"] / max_cpu_energy_consumption
                    normalized_model_energy_consumption[comb_id]["gpu"] = model_energy_consumption[comb_id]["gpu"] / max_gpu_energy_consumption
                
                # multiply normalized energy consumption by 0.5
                weighted_normalized_model_energy_consumption = deepcopy(normalized_model_energy_consumption)
                
                for comb_id in model_energy_consumption:
                    weighted_normalized_model_energy_consumption[comb_id]["cpu"] = normalized_model_energy_consumption[comb_id]["cpu"] * 0.5 # TODO: parameterize the weight
                    weighted_normalized_model_energy_consumption[comb_id]["gpu"] = normalized_model_energy_consumption[comb_id]["gpu"] * 0.5
                                    
                # Read all evaluation nas test results files
                evaluation_nas_test_results_files = glob.glob("poc_energy_efficiency_crypto/NAS/crypto_spider_5g_fcnn_optimized_benchmark_evaluation_nas_test_results_*.pkl")
                measured_performance = {}
                
                for evaluation_nas_test_results_file in evaluation_nas_test_results_files:
                    comb_id = evaluation_nas_test_results_file.split("_")[-2] + "_" + evaluation_nas_test_results_file.split("_")[-1].split(".")[0]
                    
                    with open(evaluation_nas_test_results_file, 'rb') as f:
                        evaluation_nas_test_results = pickle.load(f)

                        # # Delete evaluation nas test results file
                        # os.remove(evaluation_nas_test_results)
                    
                    # Get measured performance
                    measured_performance[comb_id] = evaluation_nas_test_results["balanced_accuracy"] # TODO: parameterize the metric
                                
                # multiply measured performance by 0.5
                weighted_measured_performance = deepcopy(measured_performance)
                
                for comb_id in measured_performance:
                    weighted_measured_performance[comb_id] = measured_performance[comb_id] * 0.5 # TODO: parameterize the weight
                
                # calculate weighted average of energy consumption and measured performance
                weighted_average_energy_consumption_and_measured_performance = {}
                
                for comb_id in model_energy_consumption:
                    weighted_average_energy_consumption_and_measured_performance[comb_id] = weighted_normalized_model_energy_consumption[comb_id]["cpu"] + weighted_measured_performance[comb_id] # TODO: parameterize the platform that is being optimized (cpu or gpu)
            
                # get best comb_id
                best_comb_id = max(weighted_average_energy_consumption_and_measured_performance, key=weighted_average_energy_consumption_and_measured_performance.get)
                logger.info(f"Best comb_id: {best_comb_id}")
                logger.info(f"Best comb_id energy consumption: {model_energy_consumption[best_comb_id]}")
                logger.info(f"Best comb_id measured performance: {measured_performance[best_comb_id]}")
                logger.info(f"Best comb_id normalized energy consumption cpu: {normalized_model_energy_consumption[best_comb_id]['cpu']}")
                logger.info(f"Best comb_id normalized energy consumption gpu: {normalized_model_energy_consumption[best_comb_id]['gpu']}")
                logger.info(f"Best comb_id weighted normalized energy consumption cpu: {weighted_normalized_model_energy_consumption[best_comb_id]['cpu']}")
                logger.info(f"Best comb_id weighted normalized energy consumption gpu: {weighted_normalized_model_energy_consumption[best_comb_id]['gpu']}")
                logger.info(f"Best comb_id weighted measured performance: {weighted_measured_performance[best_comb_id]}")
                logger.info(f"Best comb_id weighted average energy consumption and measured performance: {weighted_average_energy_consumption_and_measured_performance[best_comb_id]}")
                
                # save best comb_id
                with open(f"poc_energy_efficiency_crypto/NAS/crypto_spider_5g_fcnn_optimized_benchmark_best_comb_id.pkl", "wb") as f:
                    pickle.dump({"best_comb_id": best_comb_id}, f)
                
        # Get nas time
        nas_time = nas_time_measure()

        # Write nas time to file
        with open(f"poc_energy_efficiency_crypto/NAS/crypto_spider_5g_fcnn_optimized_benchmark_nas_time.pkl", "wb") as f:
            pickle.dump({"nas_time": nas_time}, f)
        


# In[ ]:


# Start the child process
p_nas = multiprocessing.Process(target=test_nas, args=("cpu",))

p_nas.start()
p_nas.join()


# In[ ]:


nas_times_list = []
nas_stats_list = []

device = "cpu"

for i in range(0, num_trials):
    logger.info(f"Trial {i}")
    
    p_nas = multiprocessing.Process(target=test_nas, args=(device,))
    
    p_nas.start()
    p_nas.join()
            
    # Read nas time from file
    with open(f"poc_energy_efficiency_crypto/NAS/crypto_spider_5g_fcnn_optimized_benchmark_nas_time.pkl", "rb") as f:
        nas_time = pickle.load(f)

    nas_time = nas_time["nas_time"]

    # # Delete nas time file
    # os.remove("poc_energy_efficiency_crypto/crypto_spider_5g_fcnn_optimized_benchmark_nas_time.pkl")
            
    # Read nas stats from file
    with open('poc_energy_efficiency_crypto/NAS/crypto_spider_5g_fcnn_optimized_benchmark_nas_stats.pkl', 'rb') as f:
        nas_stats = pickle.load(f)
        
    # # Delete nas stats file
    # os.remove("poc_energy_efficiency_crypto/crypto_spider_5g_fcnn_optimized_benchmark_nas_stats.pkl")

    logger.info("Trial {} - NAS time: {:.2f}s".format(i, nas_time))

    # Save NAS times
    nas_times_list.append(nas_time)

    # Save NAS stats
    nas_stats_list.append(nas_stats)

average_nas_time = np.mean(nas_times_list)
std_dev_nas_time = np.std(nas_times_list)
max_nas_time = np.max(nas_times_list)

# Time spent on NAS
logger.info("Average NAS time: {}".format(np.round(average_nas_time, 2)))
logger.info("Standard deviation of NAS time: {}".format(np.round(std_dev_nas_time, 2)))
logger.info("Max. NAS time: {}".format(np.round(max_nas_time, 2)))

# Get average NAS metrics
average_nas_stats_list = get_average_stats(nas_stats_list, average_nas_time)
std_dev_nas_stats_list = get_std_dev_stats(nas_stats_list, std_dev_nas_time)
max_nas_stats_list = get_max_stats(nas_stats_list, max_nas_time)

# Save NAS metrics to dataframe
stats = nas_stats_list[0][0].keys()

averag_nas_stats_names = [f"average_nas_{stat}" for stat in stats]
std_dev_nas_stats_names = [f"std_dev_nas_{stat}" for stat in stats]
max_nas_stats_names = [f"max_nas_{stat}" for stat in stats]

df_nas_times_columns = ["experiment", "device", "average_nas_time", "std_dev_nas_time", "max_nas_time"]
df_nas_times = pd.DataFrame(columns=df_nas_times_columns)

experiment = "nas"

nas_times_row = [experiment, device, average_nas_time, std_dev_nas_time, max_nas_time]
df_nas_times.loc[0] = nas_times_row

df_nas_stats_columns = ["experiment", "device", "snapshot", *averag_nas_stats_names, *std_dev_nas_stats_names, *max_nas_stats_names]
df_nas_stats = pd.DataFrame(columns=df_nas_stats_columns)

for index, _snapshot in enumerate(average_nas_stats_list):
    row = np.array([experiment, device, index])
    row = np.append(row, average_nas_stats_list[index])
    row = np.append(row, std_dev_nas_stats_list[index])
    row = np.append(row, max_nas_stats_list[index])

    df_nas_stats.loc[index] = row

display(df_nas_times)
display(df_nas_stats)

average_nas_time = df_nas_times["average_nas_time"].values[0]
global_average_nas_cpu_power_draw = df_nas_stats["average_nas_cpu_power_draw"].astype(float).mean()

print(f"Average NAS time: {average_nas_time}")
print(f"Global average NAS CPU power draw: {global_average_nas_cpu_power_draw}")


# In[ ]:


test_nas("cpu")


# In[ ]:


from sklearn.ensemble import RandomForestClassifier

# balance the number of samples in each class
def balance_classes(data, labels):
    # count number of samples in each class
    unique, counts = np.unique(labels, return_counts=True)
    counts_dict = dict(zip(unique, counts))
    print("Class counts:", counts_dict)
    
    # find the class with the most samples
    max_class = max(counts_dict, key=counts_dict.get)
    max_class_count = counts_dict[max_class]
    print("Class with the most samples:", max_class)
    
    # find the class with the least samples
    min_class = min(counts_dict, key=counts_dict.get)
    min_class_count = counts_dict[min_class]
    print("Class with the least samples:", min_class)
    
    # find the difference between the number of samples in the two classes
    diff = max_class_count - min_class_count
    print("Difference between the number of samples in the two classes:", diff)
    
    # find the indices of the samples in the class with the least samples
    indices = np.where(labels == min_class)[0]
    # randomly select the same number of samples from the class with the most samples
    selected_indices = np.random.choice(np.where(labels == max_class)[0], diff, replace=False)
    # combine the indices of the two classes
    combined_indices = np.concatenate((indices, selected_indices))
    # shuffle the indices
    shuffled_indices = np.random.permutation(combined_indices)
    # return the balanced data and labels
    
    return data[shuffled_indices], labels[shuffled_indices]

balanced_data, balanced_labels = balance_classes(data_transformed, df_train['tag'].values)

rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=0)
# rf.fit(data_transformed, df_train['tag'].values)
rf.fit(balanced_data, balanced_labels)

test_data_transformed = standard.transform(df_test[gf])
predictions = rf.predict(test_data_transformed)

accuracy = accuracy_score(df_test["tag"], predictions)
balanced_accuracy = balanced_accuracy_score(df_test["tag"], predictions)
f1 = f1_score(df_test["tag"], predictions, average="weighted")

print(f"Accuracy: {accuracy}")
print(f"Balanced accuracy: {balanced_accuracy}")
print(f"F1 score: {f1}")


# In[ ]:


teacher = baseline_model(input_dim=len(gf), n_output=2)
teacher, history = train_model(teacher)


# In[ ]:


student = small_model(input_dim=len(gf), n_output=2)

# Initialize and compile distiller
distiller = Distiller(student=student, teacher=teacher)
distiller.compile(
    optimizer=keras.optimizers.Adam(),
    metrics=[keras.metrics.CategoricalAccuracy()],
    student_loss_fn=keras.losses.CategoricalCrossentropy(from_logits=False),
    distillation_loss_fn=keras.losses.KLDivergence(),
    alpha=0.1,
    temperature=1,
)

es = EarlyStopping(
    monitor="val_student_loss",
    mode="min",
    patience=es_patience,
    restore_best_weights=es_restore_best_weights,
)
history = distiller.fit(
    data_transformed,
    pd.get_dummies(df_train["tag"]),
    validation_split=validation_split,
    epochs=epochs,
    batch_size=batch_size,
    callbacks=[es],
    verbose=False,
)

test_data_transformed = standard.transform(df_test[gf])
predictions = distiller.student.predict(test_data_transformed)
balanced_accuracy = balanced_accuracy_score(df_test["tag"], np.argmax(predictions, axis=1))
print(f"Balanced accuracy: {balanced_accuracy}")


# In[ ]:


# Initialize and compile distiller
distiller = Distiller(student=student, teacher=teacher)
distiller.compile(
    optimizer=keras.optimizers.Adam(),
    metrics=[keras.metrics.CategoricalAccuracy()],
    student_loss_fn=keras.losses.CategoricalCrossentropy(from_logits=False),
    distillation_loss_fn=keras.losses.KLDivergence(),
    alpha=0.1,
    temperature=3,
)

es = EarlyStopping(
    monitor="val_student_loss",
    mode="min",
    patience=es_patience,
    restore_best_weights=es_restore_best_weights,
)
history = distiller.fit(
    data_transformed,
    pd.get_dummies(df_train["tag"]),
    validation_split=validation_split,
    epochs=epochs,
    batch_size=batch_size,
    callbacks=[es],
    verbose=1,
)


# In[ ]:


average_nas_exp_duration = df_nas_times[f"average_{experiment}_time"]
df_nas_stats["total_average_cpu_energy_consumption"] = df_nas_stats[f"average_nas_cpu_power_draw"] * average_nas_exp_duration
df_nas_stats["total_average_gpu_energy_consumption"] = df_nas_stats[f"average_nas_gpu_power_draw"] * average_nas_exp_duration


