# #### Pruning

# In[50]:


baseline = baseline_model(input_dim=len(gf), n_output=2)
baseline, history = train_model(baseline)


# In[58]:


from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV
import tensorflow as tf
import tensorflow_model_optimization as tfmot
import numpy as np
import pandas as pd

class PolynomialDecayPruningEstimator(BaseEstimator):
    def __init__(self, batch_size=256, epochs=10, initial_sparsity=0.50, final_sparsity=0.80):
        self.batch_size = batch_size
        self.epochs = epochs
        self.initial_sparsity = initial_sparsity
        self.final_sparsity = final_sparsity
    
    def fit(self, X, y):
        # Fine-tune pretrained model with pruning aware training
        prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

        num_samples = len(X)
        end_step = np.ceil(num_samples / self.batch_size).astype(np.int32) * self.epochs

        pruning_params = {
            "pruning_schedule": tfmot.sparsity.keras.PolynomialDecay(
                initial_sparsity=self.initial_sparsity, final_sparsity=self.final_sparsity, begin_step=0, end_step=end_step
            )
        }

        model_for_pruning = prune_low_magnitude(baseline, **pruning_params)

        model_for_pruning.compile(
            optimizer="adam", loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False), metrics=["accuracy"]
        )

        callbacks = [
            tfmot.sparsity.keras.UpdatePruningStep(),
        ]

        self.history_ = model_for_pruning.fit(X, y, validation_split=0.1, epochs=self.epochs, batch_size=self.batch_size, callbacks=callbacks, verbose=0)

        self.pruned_model_ = tfmot.sparsity.keras.strip_pruning(model_for_pruning)

        # convert to tflite
        converter = tf.lite.TFLiteConverter.from_keras_model(self.pruned_model_)
        converter.optimizations = [tf.lite.Optimize.EXPERIMENTAL_SPARSITY, tf.lite.Optimize.DEFAULT]
        self.pruned_tflite_model_ = converter.convert()

        # Save the model to disk
        self.pruned_tflite_model_file_ = 'model.tflite'

        with open(self.pruned_tflite_model_file_, 'wb') as f:
            f.write(self.pruned_tflite_model_)

        return self
    
    def score(self, X, y):
        # Load model
        interpreter = tf.lite.Interpreter(model_path=self.pruned_tflite_model_file_)
        interpreter.allocate_tensors()

        # Load test set
        # test_data_transformed = standard.transform(df_test[gf])
        
        # get input shape
        input_shape = interpreter.get_input_details()[0]["shape"]
        logger.info(f"Input shape: {input_shape}")
        
        # get output shape
        output_shape = interpreter.get_output_details()[0]["shape"]
        logger.info(f"Output shape: {output_shape}")
        
        # transform data to the expected tensor type
        input_details = interpreter.get_input_details()[0]
        dtype = input_details["dtype"]    
        input_data = np.array(X, dtype=dtype)
        
        # reshape model input
        batch_size = 256
        
        input_details = interpreter.get_input_details()[0]
        interpreter.resize_tensor_input(input_details['index'], (batch_size, input_data.shape[1]))
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()[0]
        output_details = interpreter.get_output_details()[0]
        
        preds = []

        # create batches of test_data_transformed
        for i in range(0, len(X), batch_size):
            batch = X[i:i+batch_size]
            # print(f"Batch {i//batch_size} has {len(batch)} elements")
            
            batch_data = np.array(batch, dtype=dtype)
            
            if len(batch) == batch_size:        
                interpreter.set_tensor(input_details['index'], batch_data)
                interpreter.invoke()
                output_data = interpreter.get_tensor(output_details['index'])
                
                preds.append(output_data)
        
        predictions = np.concatenate(preds)
        
        # take the labels of the test set
        test_labels = y.values[:len(predictions)]
        
        balanced_accuracy = balanced_accuracy_score(np.argmax(y, axis=1), np.argmax(predictions, axis=1))
        
        return balanced_accuracy
    
    def evaluate(self, X, y):
        # Load model
        interpreter = tf.lite.Interpreter(model_path=self.pruned_tflite_model_file_)
        interpreter.allocate_tensors()

        # get input shape
        input_shape = interpreter.get_input_details()[0]["shape"]
        logger.info(f"Input shape: {input_shape}")

        # get output shape
        output_shape = interpreter.get_output_details()[0]["shape"]
        logger.info(f"Output shape: {output_shape}")

        # transform data to the expected tensor type
        input_details = interpreter.get_input_details()[0]
        dtype = input_details["dtype"]
        input_data = np.array(X, dtype=dtype)

        # reshape model input
        batch_size = 256

        input_details = interpreter.get_input_details()[0]
        interpreter.resize_tensor_input(
            input_details["index"], (batch_size, input_data.shape[1])
        )
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()[0]
        output_details = interpreter.get_output_details()[0]

        preds = []

        # create batches of test_data_transformed
        for i in range(0, len(X), batch_size):
            batch = X[i : i + batch_size]
            # print(f"Batch {i//batch_size} has {len(batch)} elements")

            batch_data = np.array(batch, dtype=dtype)

            if len(batch) == batch_size:
                interpreter.set_tensor(input_details["index"], batch_data)
                interpreter.invoke()
                output_data = interpreter.get_tensor(output_details["index"])

                preds.append(output_data)
                
        predictions = np.concatenate(preds)

        # take the labels of the test set
        test_labels = y.values[: len(predictions)]

        balanced_accuracy = balanced_accuracy_score(
            np.argmax(test_labels, axis=1), np.argmax(predictions, axis=1)
        )

        accuracy = accuracy_score(np.argmax(test_labels, axis=1), np.argmax(predictions, axis=1))
        balanced_accuracy = balanced_accuracy_score(np.argmax(test_labels, axis=1), np.argmax(predictions, axis=1))
        f1 = f1_score(np.argmax(test_labels, axis=1), np.argmax(predictions, axis=1), average="weighted")
        
        accuracy = round(accuracy, 3)
        balanced_accuracy = round(balanced_accuracy, 3)
        f1 = round(f1, 3)
        
        print(f"Accuracy: {accuracy}")
        print(f"Balanced accuracy: {balanced_accuracy}")
        print(f"F1 score: {f1}")


# In[ ]:


with tf.device('/device:CPU:0'):
    param_grid = {'batch_size': [256, 512, 1024, 2048], 'epochs': [10, 20, 30], 'initial_sparsity': [0.50, 0.60, 0.70, 0.80], 'final_sparsity': [0.60, 0.70, 0.80, 0.90]}

    grid_search = GridSearchCV(estimator=PolynomialDecayPruningEstimator(), param_grid=param_grid, cv=3, verbose=2)
    grid_search.fit(data_transformed, pd.get_dummies(df_train["tag"]))

    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best score: {grid_search.best_score_}")


# In[59]:


with tf.device('/device:CPU:0'):
    pe = PolynomialDecayPruningEstimator(batch_size=1024, epochs=10, initial_sparsity=0.70, final_sparsity=0.90)
    pe.fit(data_transformed, pd.get_dummies(df_train["tag"]))
    pe.evaluate(test_data_transformed, pd.get_dummies(df_test["tag"]))


# In[ ]:


from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV
import tensorflow as tf
import tensorflow_model_optimization as tfmot
import numpy as np
import pandas as pd

class ConstantSparsityPruningEstimator(BaseEstimator):
    def __init__(self, batch_size=256, epochs=10, target_sparsity=0.80):
        self.batch_size = batch_size
        self.epochs = epochs
        self.target_sparsity = target_sparsity
    
    def fit(self, X, y):
        # Fine-tune pretrained model with pruning aware training
        prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

        num_samples = len(X)
        end_step = np.ceil(num_samples / self.batch_size).astype(np.int32) * self.epochs

        pruning_params = {
            "pruning_schedule": tfmot.sparsity.keras.ConstantSparsity(
                target_sparsity=self.target_sparsity, begin_step=0, end_step=end_step,
            )
        }

        model_for_pruning = prune_low_magnitude(baseline, **pruning_params)

        model_for_pruning.compile(
            optimizer="adam", loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False), metrics=["accuracy"]
        )

        callbacks = [
            tfmot.sparsity.keras.UpdatePruningStep(),
        ]

        self.history_ = model_for_pruning.fit(X, y, validation_split=0.1, epochs=self.epochs, batch_size=self.batch_size, callbacks=callbacks, verbose=0)

        self.pruned_model_ = tfmot.sparsity.keras.strip_pruning(model_for_pruning)

        # convert to tflite
        converter = tf.lite.TFLiteConverter.from_keras_model(self.pruned_model_)
        converter.optimizations = [tf.lite.Optimize.EXPERIMENTAL_SPARSITY, tf.lite.Optimize.DEFAULT]
        self.pruned_tflite_model_ = converter.convert()

        # Save the model to disk
        self.pruned_tflite_model_file_ = 'model.tflite'

        with open(self.pruned_tflite_model_file_, 'wb') as f:
            f.write(self.pruned_tflite_model_)

        return self
    
    def score(self, X, y):
        # Load model
        interpreter = tf.lite.Interpreter(model_path=self.pruned_tflite_model_file_)
        interpreter.allocate_tensors()

        # Load test set
        # test_data_transformed = standard.transform(df_test[gf])
        
        # get input shape
        input_shape = interpreter.get_input_details()[0]["shape"]
        logger.info(f"Input shape: {input_shape}")
        
        # get output shape
        output_shape = interpreter.get_output_details()[0]["shape"]
        logger.info(f"Output shape: {output_shape}")
        
        # transform data to the expected tensor type
        input_details = interpreter.get_input_details()[0]
        dtype = input_details["dtype"]    
        input_data = np.array(X, dtype=dtype)
        
        # reshape model input
        batch_size = 256
        
        input_details = interpreter.get_input_details()[0]
        interpreter.resize_tensor_input(input_details['index'], (batch_size, input_data.shape[1]))
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()[0]
        output_details = interpreter.get_output_details()[0]
        
        preds = []

        # create batches of test_data_transformed
        for i in range(0, len(X), batch_size):
            batch = X[i:i+batch_size]
            # print(f"Batch {i//batch_size} has {len(batch)} elements")
            
            batch_data = np.array(batch, dtype=dtype)
            
            if len(batch) == batch_size:        
                interpreter.set_tensor(input_details['index'], batch_data)
                interpreter.invoke()
                output_data = interpreter.get_tensor(output_details['index'])
                
                preds.append(output_data)
        
        predictions = np.concatenate(preds)
        
        # take the labels of the test set
        test_labels = y.values[:len(predictions)]
        
        balanced_accuracy = balanced_accuracy_score(np.argmax(y, axis=1), np.argmax(predictions, axis=1))
        
        return balanced_accuracy


# In[ ]:


with tf.device('/device:CPU:0'):
    param_grid = {'batch_size': [256, 512, 1024, 2048], 'epochs': [10, 20, 30], 'target_sparsity': [0.5, 0.60, 0.70, 0.80, 0.90]}

    grid_search = GridSearchCV(estimator=ConstantSparsityPruningEstimator(), param_grid=param_grid, cv=3, verbose=2)
    grid_search.fit(data_transformed, pd.get_dummies(df_train["tag"]))

    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best score: {grid_search.best_score_}")


# In[ ]:


print(f"Best parameters: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_}")


# In[ ]:


import tensorflow as tf
import tensorflow_model_optimization as tfmot
import numpy as np
import pandas as pd

class PolynomialDecayPruningEstimator:
    def __init__(self, model=None, batch_size=256, epochs=10, initial_sparsity=0.50, final_sparsity=0.80):
        self.model = model
        self.batch_size = batch_size
        self.epochs = epochs
        self.initial_sparsity = initial_sparsity
        self.final_sparsity = final_sparsity
    
    def fit(self, X, y):
        # Fine-tune pretrained model with pruning aware training
        prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

        num_samples = len(X)
        end_step = np.ceil(num_samples / self.batch_size).astype(np.int32) * self.epochs

        pruning_params = {
            "pruning_schedule": tfmot.sparsity.keras.PolynomialDecay(
                initial_sparsity=self.initial_sparsity, final_sparsity=self.final_sparsity, begin_step=0, end_step=end_step
            )
        }

        model_for_pruning = prune_low_magnitude(self.model, **pruning_params)

        model_for_pruning.compile(
            optimizer="adam", loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False), metrics=["accuracy"]
        )

        callbacks = [
            tfmot.sparsity.keras.UpdatePruningStep(),
        ]

        self.history_ = model_for_pruning.fit(X, y, validation_split=0.1, epochs=self.epochs, batch_size=self.batch_size, callbacks=callbacks, verbose=0)

        self.pruned_model_ = tfmot.sparsity.keras.strip_pruning(model_for_pruning)

        # convert to tflite
        converter = tf.lite.TFLiteConverter.from_keras_model(self.pruned_model_)
        converter.optimizations = [tf.lite.Optimize.EXPERIMENTAL_SPARSITY, tf.lite.Optimize.DEFAULT]
        self.pruned_tflite_model_ = converter.convert()

        # Save the model to disk
        self.pruned_tflite_model_file_ = 'model.tflite'

        with open(self.pruned_tflite_model_file_, 'wb') as f:
            f.write(self.pruned_tflite_model_)

        return self
    
    def score(self, X, y):
        # Load model
        interpreter = tf.lite.Interpreter(model_path=self.pruned_tflite_model_file_)
        interpreter.allocate_tensors()
        
        # get input shape
        input_shape = interpreter.get_input_details()[0]["shape"]
        logger.info(f"Input shape: {input_shape}")
        
        # get output shape
        output_shape = interpreter.get_output_details()[0]["shape"]
        logger.info(f"Output shape: {output_shape}")
        
        # transform data to the expected tensor type
        input_details = interpreter.get_input_details()[0]
        dtype = input_details["dtype"]    
        input_data = np.array(X, dtype=dtype)
        
        # reshape model input
        batch_size = 256
        
        input_details = interpreter.get_input_details()[0]
        interpreter.resize_tensor_input(input_details['index'], (batch_size, input_data.shape[1]))
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()[0]
        output_details = interpreter.get_output_details()[0]
        
        preds = []

        # create batches of test_data_transformed
        for i in range(0, len(X), batch_size):
            batch = X[i:i+batch_size]
            
            batch_data = np.array(batch, dtype=dtype)
            
            if len(batch) == batch_size:        
                interpreter.set_tensor(input_details['index'], batch_data)
                interpreter.invoke()
                output_data = interpreter.get_tensor(output_details['index'])
                
                preds.append(output_data)
        
        predictions = np.concatenate(preds)
        
        # take the labels of the test set
        test_labels = y.values[:len(predictions)]
        
        balanced_accuracy = balanced_accuracy_score(np.argmax(test_labels, axis=1), np.argmax(predictions, axis=1))
        
        return balanced_accuracy
    
    def predict(self, X):
        # Load model
        interpreter = tf.lite.Interpreter(model_path=self.pruned_tflite_model_file_)
        interpreter.allocate_tensors()
        
        # get input shape
        input_shape = interpreter.get_input_details()[0]["shape"]
        logger.info(f"Input shape: {input_shape}")
        
        # get output shape
        output_shape = interpreter.get_output_details()[0]["shape"]
        logger.info(f"Output shape: {output_shape}")
        
        # transform data to the expected tensor type
        input_details = interpreter.get_input_details()[0]
        dtype = input_details["dtype"]    
        input_data = np.array(X, dtype=dtype)
        
        # reshape model input
        batch_size = 256
        
        input_details = interpreter.get_input_details()[0]
        interpreter.resize_tensor_input(input_details['index'], (batch_size, input_data.shape[1]))
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()[0]
        output_details = interpreter.get_output_details()[0]
        
        preds = []

        # create batches of test_data_transformed
        for i in range(0, len(X), batch_size):
            batch = X[i:i+batch_size]
            
            batch_data = np.array(batch, dtype=dtype)
            
            if len(batch) == batch_size:        
                interpreter.set_tensor(input_details['index'], batch_data)
                interpreter.invoke()
                output_data = interpreter.get_tensor(output_details['index'])
                
                preds.append(output_data)
        
        predictions = np.concatenate(preds)
        
        return predictions


# In[ ]:


def test_pruning(device="CPU"):
    baseline = baseline_model(input_dim=len(gf), n_output=2)
    baseline, _history = train_model(baseline)

    num_trials = 5
    
    logger.info("Starting test_pruning")
    
    pid = os.getpid()
    
    model_energy_consumption = {}
    
    Path(f"poc_energy_efficiency_crypto/Pruning").mkdir(parents=True, exist_ok=True)
    
    with tf.device(device):
        with StatsCollectionManager(directory="Pruning", test="pruning", sampling_rate=stats_sampling_rate, pid=pid) as pruning_scm:
            with measure_time() as pruning_time_measure:
                logger.info("Applying Pruning")
        
                # find the student model structure by searching for the model
                param_grid = {'batch_size': [256, 512, 1024, 2048], 'epochs': [10, 20, 30], 'initial_sparsity': [0.50, 0.60, 0.70, 0.80], 'final_sparsity': [0.60, 0.70, 0.80, 0.90]}
                # param_grid = {'batch_size': [256], 'epochs': [10], 'initial_sparsity': [0.50], 'final_sparsity': [0.60]}
                batch_size_grid = param_grid['batch_size']
                epochs_grid = param_grid['epochs']
                initial_sparsity_grid = param_grid['initial_sparsity']
                final_sparsity_grid = param_grid['final_sparsity']

                # make all possible combinations of the parameters
                combinations = list(itertools.product(batch_size_grid, epochs_grid, initial_sparsity_grid, final_sparsity_grid))
                    
                for comb_id, comb in enumerate(combinations):
                    batch_size = comb[0]
                    epochs = comb[1]
                    initial_sparsity = comb[2]
                    final_sparsity = comb[3]
                    
                    logger.info(f"Parameters: batch_size={batch_size}, epochs={epochs}, initial_sparsity={initial_sparsity}, final_sparsity={final_sparsity}")
                    
                    estimator = PolynomialDecayPruningEstimator(model=baseline, batch_size=batch_size, epochs=epochs, initial_sparsity=initial_sparsity, final_sparsity=final_sparsity)
                    estimator.fit(data_transformed, pd.get_dummies(df_train["tag"]))
                        
                    for i in range(0, num_trials):
                        logger.info(f"Trial Evaluation: {i}")
                    
                        # evaluate the model
                        data_transformed_test = standard.transform(df_test[gf])
                        
                        with StatsCollectionManager(directory="Pruning", test=f"evaluation_pruning_{comb_id}_{i}", sampling_rate=stats_sampling_rate, pid=pid) as evaluation_pruning_scm:
                            with measure_time() as evaluation_pruning_time_measure:
                                predictions = estimator.predict(data_transformed_test)
                        
                        evaluation_pruning_time = evaluation_pruning_time_measure()
                        
                        # Write evaluation pruning time to file
                        with open(f"poc_energy_efficiency_crypto/Pruning/crypto_spider_5g_fcnn_optimized_benchmark_evaluation_pruning_time_{comb_id}_{i}.pkl", "wb") as f:
                            pickle.dump({"evaluation_pruning_time": evaluation_pruning_time}, f)
                            
                    if ml_task == "binary_classification":
                        predictions = np.argmax(predictions, axis=1)
                        
                        test_labels = df_test["tag"].values[:len(predictions)]
                        
                        accuracy = accuracy_score(test_labels, predictions)
                        f1 = f1_score(test_labels, predictions, average="weighted")
                        auc = roc_auc_score(test_labels, predictions)
                        recall = recall_score(test_labels, predictions, average="weighted")
                        precision = precision_score(test_labels, predictions, average="weighted")
                        balanced_accuracy = balanced_accuracy_score(test_labels, predictions)
                        matthews = matthews_corrcoef(test_labels, predictions)

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
                        mae = mean_absolute_error(test_labels, predictions)
                        mse = mean_squared_error(test_labels, predictions)
                        mape = mean_absolute_percentage_error(test_labels, predictions)
                        smape = 1/len(test_labels) * np.sum(2 * np.abs(predictions - test_labels) / (np.abs(predictions) + np.abs(test_labels)))
                        
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
                        
                    with open(f"poc_energy_efficiency_crypto/Pruning/crypto_spider_5g_fcnn_optimized_benchmark_evaluation_pruning_test_results_{comb_id}.pkl", "wb") as f:
                        pickle.dump(test_results, f)
                                
                    evaluation_pruning_times_list = []
                    evaluation_pruning_stats_list = []
                    
                    for i in range(0, num_trials):
                        logger.info(f"Trial Evaluation: {i}. Comb ID: {comb_id}")
                            
                        # Read pruning time from file
                        with open(f"poc_energy_efficiency_crypto/Pruning/crypto_spider_5g_fcnn_optimized_benchmark_evaluation_pruning_time_{comb_id}_{i}.pkl", "rb") as f:
                            evaluation_pruning_time = pickle.load(f)

                        evaluation_pruning_time = evaluation_pruning_time["evaluation_pruning_time"]
                        evaluation_pruning_time = float(evaluation_pruning_time)
                                
                        # Read pruning stats from file
                        with open(f'poc_energy_efficiency_crypto/Pruning/crypto_spider_5g_fcnn_optimized_benchmark_evaluation_pruning_{comb_id}_{i}_stats.pkl', 'rb') as f:
                            evaluation_pruning_stats = pickle.load(f)
                        
                        logger.info("Trial {}. Comb ID {}. Evaluation Pruning time: {}".format(i, comb_id, evaluation_pruning_time))

                        # Save Pruning times
                        evaluation_pruning_times_list.append(evaluation_pruning_time)

                        # Save Pruning stats
                        evaluation_pruning_stats_list.append(evaluation_pruning_stats)

                    average_evaluation_pruning_time = np.mean(evaluation_pruning_times_list)
                    std_dev_evaluation_pruning_time = np.std(evaluation_pruning_times_list)
                    max_evaluation_pruning_time = np.max(evaluation_pruning_times_list)

                    # Time spent on Pruning
                    logger.info(f"Comb ID: {comb_id}. " + "Average Pruning evaluation time: {}".format(np.round(average_evaluation_pruning_time, 2)))
                    logger.info(f"Comb ID: {comb_id}. " + "Standard deviation of Pruning evaluation time: {}".format(np.round(std_dev_evaluation_pruning_time, 2)))
                    logger.info(f"Comb ID: {comb_id}. " + "Max. Pruning evaluation time: {}".format(np.round(max_evaluation_pruning_time, 2)))

                    # Get average Pruning metrics
                    average_evaluation_pruning_stats_list = get_average_stats(evaluation_pruning_stats_list, average_evaluation_pruning_time)
                    std_dev_evaluation_pruning_stats_list = get_std_dev_stats(evaluation_pruning_stats_list, std_dev_evaluation_pruning_time)
                    max_evaluation_pruning_stats_list = get_max_stats(evaluation_pruning_stats_list, max_evaluation_pruning_time)
                    
                    # Save Pruning metrics to dataframe
                    stats = evaluation_pruning_stats_list[0][0].keys()

                    averag_evaluation_pruning_stats_names = [f"average_evaluation_pruning_{stat}" for stat in stats]
                    std_dev_evaluation_pruning_stats_names = [f"std_dev_evaluation_pruning_{stat}" for stat in stats]
                    max_evaluation_pruning_stats_names = [f"max_evaluation_pruning_{stat}" for stat in stats]

                    df_evaluation_pruning_times_columns = ["experiment", "device", "average_evaluation_pruning_time", "std_dev_evaluation_pruning_time", "max_evaluation_pruning_time"]
                    df_evaluation_pruning_times = pd.DataFrame(columns=df_evaluation_pruning_times_columns)
                    
                    experiment = f"evaluation_pruning_{comb_id}"
                    evaluation_pruning_times_row = [experiment, device, average_evaluation_pruning_time, std_dev_evaluation_pruning_time, max_evaluation_pruning_time]
                    df_evaluation_pruning_times.loc[0] = evaluation_pruning_times_row

                    df_evaluation_pruning_stats_columns = ["experiment", "device", "snapshot", *averag_evaluation_pruning_stats_names, *std_dev_evaluation_pruning_stats_names, *max_evaluation_pruning_stats_names]
                    df_evaluation_pruning_stats = pd.DataFrame(columns=df_evaluation_pruning_stats_columns)
                    
                    for index, _snapshot in enumerate(average_evaluation_pruning_stats_list):
                        row = np.array([experiment, device, index])
                        row = np.append(row, average_evaluation_pruning_stats_list[index])
                        row = np.append(row, std_dev_evaluation_pruning_stats_list[index])
                        row = np.append(row, max_evaluation_pruning_stats_list[index])

                        df_evaluation_pruning_stats.loc[index] = row
                        
                    assert df_evaluation_pruning_times.shape[0] == 1
                    assert df_evaluation_pruning_stats.shape[0] == len(average_evaluation_pruning_stats_list)
                    assert df_evaluation_pruning_stats.shape[1] == len(df_evaluation_pruning_stats_columns)
                    
                    display(df_evaluation_pruning_times)
                    display(df_evaluation_pruning_stats)

                    # calculate the average of the df_evaluation_pruning_stats average_evaluation_pruning_cpu_power_draw and average_evaluation_pruning_gpu_power_draw columns
                    global_average_average_evaluation_pruning_cpu_power_draw = df_evaluation_pruning_stats[f"average_evaluation_pruning_cpu_power_draw"].astype(float).mean()
                    global_average_average_evaluation_pruning_gpu_power_draw = df_evaluation_pruning_stats[f"average_evaluation_pruning_gpu_power_draw"].astype(float).mean()
                    
                    # check that both columns have a single scalar value
                    assert np.isscalar(global_average_average_evaluation_pruning_cpu_power_draw)
                    assert np.isscalar(global_average_average_evaluation_pruning_gpu_power_draw)
                    
                    average_evaluation_pruning_exp_duration = df_evaluation_pruning_times[f"average_evaluation_pruning_time"].values[0]
                    
                    logger.info("Average evaluation Pruning experiment duration: {}".format(average_evaluation_pruning_exp_duration))
                    logger.info("Average evaluation Pruning CPU power draw: {}".format(global_average_average_evaluation_pruning_cpu_power_draw))
                    logger.info("Average evaluation Pruning GPU power draw: {}".format(global_average_average_evaluation_pruning_gpu_power_draw))
                    
                    total_average_cpu_energy_consumption = global_average_average_evaluation_pruning_cpu_power_draw * average_evaluation_pruning_exp_duration
                    total_average_gpu_energy_consumption = global_average_average_evaluation_pruning_gpu_power_draw * average_evaluation_pruning_exp_duration
                    
                    model_energy_consumption[comb_id] = {
                        "cpu": total_average_cpu_energy_consumption,
                        "gpu": total_average_gpu_energy_consumption
                    }
                    
                    logger.info("Model energy consumption:\n{}".format(model_energy_consumption))
                    
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
                                    
                # Read all evaluation pruning test results files
                evaluation_pruning_test_results_files = glob.glob("poc_energy_efficiency_crypto/Pruning/crypto_spider_5g_fcnn_optimized_benchmark_evaluation_pruning_test_results_*.pkl")
                measured_performance = {}
                
                for evaluation_pruning_test_results_file in evaluation_pruning_test_results_files:
                    # comb_id = evaluation_pruning_test_results_file.split("_")[-2] + "_" + evaluation_pruning_test_results_file.split("_")[-1].split(".")[0]
                    comb_id = int(evaluation_pruning_test_results_file.split("_")[-1].split(".")[0])
                    
                    with open(evaluation_pruning_test_results_file, 'rb') as f:
                        evaluation_pruning_test_results = pickle.load(f)
                    
                    # Get measured performance
                    measured_performance[comb_id] = evaluation_pruning_test_results["balanced_accuracy"] # TODO: parameterize the metric
                                
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
                with open(f"poc_energy_efficiency_crypto/Pruning/crypto_spider_5g_fcnn_optimized_benchmark_best_comb_id.pkl", "wb") as f:
                    pickle.dump({"best_comb_id": best_comb_id}, f)
                
        # Get pruning time
        pruning_time = pruning_time_measure()

        # Write pruning time to file
        with open(f"poc_energy_efficiency_crypto/Pruning/crypto_spider_5g_fcnn_optimized_benchmark_pruning_time.pkl", "wb") as f:
            pickle.dump({"pruning_time": pruning_time}, f)
        


# In[ ]:


# Start the child process
p_nas = multiprocessing.Process(target=test_pruning, args=("cpu",))

p_nas.start()
p_nas.join()


