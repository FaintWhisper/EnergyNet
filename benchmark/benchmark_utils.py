import os
import time
import gzip
import pickle
import numpy as np
import tensorflow as tf
import onnx
from contextlib import contextmanager
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, balanced_accuracy_score, matthews_corrcoef

@contextmanager
def measure_time() -> float:
    start = time.perf_counter()
    yield lambda: time.perf_counter() - start

class StatsCollectionManager():
    def __init__(self, test, sampling_rate=0.1, pid=None, directory=None):
        self.test = test
        self.sampling_rate = sampling_rate
        self.pid = pid
        self.proc = None
        
        if directory is None:
            self.directory = f"poc_energy_efficiency_crypto/"
        else:
            self.directory = f"poc_energy_efficiency_crypto/{directory}/"

        print("Starting stats collection for test: {}".format(test))
 
    def __enter__(self):
        self.proc = get_stats_background(test=self.test, sampling_rate=self.sampling_rate, pid=self.pid, directory=self.directory)

        while True:
            # check if file stats.txt exists
            if os.path.exists(f"started_{self.test}.txt"):
                print("\nStats collection started")
                os.remove(f"started_{self.test}.txt")
                break
 
    def __exit__(self, exc_type, exc_value, exc_traceback):
        # write file stop.txt to signal to the background process to stop
        with open(f"stop_{self.test}.txt", "w") as f:
            print("Stopping stats collection")
            f.write("STOP")

        while True:
            if os.path.isfile(f"{self.directory}/crypto_spider_5g_fcnn_optimized_benchmark_{self.test}_stats.pkl"):
                print("Stats file found")
                print(self.test)

                break

def perform_inference(model):
    model.predict(data_transformed)
    
def perform_inference_onnx(sess):
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    sess.run([output_name], {input_name: data_transformed.astype(np.float32)})
    
def perform_inference_tflite(interpreter, batch_size):
    # check input details and convert data to the right type
    input_details = interpreter.get_input_details()
    print(input_details)
    dtype = input_details[0]['dtype']
    input_data = np.array(data_transformed, dtype=dtype)
    
    input_details = interpreter.get_input_details()[0]
    interpreter.resize_tensor_input(input_details['index'], (batch_size, input_data.shape[1]))
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    
    # create batches of data_transformed
    batch_size = batch_size
    
    preds = []

    for i in range(0, len(data_transformed), batch_size):
        batch = data_transformed[i:i+batch_size]
        # print(f"Batch {i//batch_size} has {len(batch)} elements")
        
        batch_data = np.array(batch, dtype=dtype)
        
        if len(batch) == batch_size:        
            interpreter.set_tensor(input_details['index'], batch_data)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details['index'])
            
            preds.append(output_data)
    
    return np.concatenate(preds)

def save_model_to_h5(keras_model, model_filename):
    keras_model.save(f"{model_filename}.h5", include_optimizer=False)
    
    return f"{model_filename}.h5"

def save_model_to_tflite(tflite_model, model_filename):
    with open(f"{model_filename}.tflite", 'wb') as f:
        f.write(tflite_model)
    
    return f"{model_filename}.tflite"

def save_model_to_onnx(onnx_model, model_filename):
    onnx.save(onnx_model, f"{model_filename}.onnx")
    
    return f"{model_filename}.onnx"
    
def gzip_model(model_path):
    with open(model_path, 'rb') as f_in, open(f"{model_path}.gz", 'wb') as f_out:
        f_out.write(gzip.compress(f_in.read()))
        
    os.remove(model_path)
    
    return f"{model_path}.gz"

def unzip_model(model_path):
    with gzip.open(model_path, 'rb') as f_in, open(model_path[:-3], 'wb') as f_out:
        f_out.write(f_in.read())
            
    return model_path[:-3]

def save_model(model, ext):
    model_filename = "model"
    
    if ext == "h5":
        model_path = save_model_to_h5(model, model_filename)
    elif ext == "onnx":
        model_path = save_model_to_onnx(model, model_filename)
    elif ext == "tflite":
        model_path = save_model_to_tflite(model, model_filename)
    else:
        raise Exception("Model format not supported")
    
    model_path = gzip_model(model_path)
    
    print(f"Model saved to {model_path}")
    
    return model_path

def load_model_from_h5(model_path) -> tf.keras.Model:
    keras_model = tf.keras.models.load_model(model_path)
    
    return keras_model

def load_model_from_tflite(model_path) -> tf.lite.Interpreter:
    tflite_model = tf.lite.Interpreter(model_path=model_path)
    tflite_model.allocate_tensors()
    
    return tflite_model

def load_model_from_onnx(model_path) -> onnx.ModelProto:
    # onnx_model = onnx.load(model_path)
    
    # return onnx_model
    
    # onnx_model = rt.InferenceSession(model_path, providers=["CUDAExecutionProvider"])
    onnx_model = rt.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    
    return onnx_model

def load_model(ext):
    model_path = f"model.{ext}.gz"
    
    model_path = unzip_model(model_path)
    
    if ext == "h5":
        model = load_model_from_h5(model_path)
    elif ext == "onnx":
        model = load_model_from_onnx(model_path)
    elif ext == "tflite":
        model = load_model_from_tflite(model_path)
    else:
        raise Exception("Model format not supported")
    
    return model

def export_keras_model_to_tflite(model):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    
    return tflite_model

def export_keras_model_to_onnx(model):
    spec = (tf.TensorSpec(shape=(1, 16), dtype=tf.float32, name='input'),)
    model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, output_path="model.onnx")
    
    return model_proto

def export_tflite_model_to_onnx(model_path):
    spec = (tf.TensorSpec(shape=(1, 16), dtype=tf.float32, name='input'),)
    model_proto, _ = tf2onnx.convert.from_tflite(model_path, output_path="model.onnx")
    
    return model_proto

def get_gzipped_model_size(ext):
  # Returns size of gzipped model, in bytes.
  if os.path.exists(f"model.{ext}.gz"):
    return os.path.getsize(f"model.{ext}.gz")
  else:
      RuntimeError(f"Model with extension \"{ext}\" not found")

def delete_model(ext):
    if os.path.exists(f"model.{ext}.gz"):
        os.remove(f"model.{ext}.gz")
    else:
        RuntimeError(f"Model with extension \"{ext}\" not found")
    
    if os.path.exists(f"model.{ext}"):
        os.remove(f"model.{ext}")
    else:
        RuntimeError(f"Model with extension \"{ext}\" not found")

def perform_evaluation_onnx(experiment, device="CPU"):
    assert device in ["CPU", "GPU"]

    logger.info(f"Evaluating {experiment} model on test set")

    # Load model
    sess = load_model(ext="onnx")
    sess.set_providers(["CPUExecutionProvider"] if device == "CPU" else ["CUDAExecutionProvider"])

    # Load test set
    test_data_transformed = standard.transform(df_test[gf])
    
    # get input shape
    input_shape = sess.get_inputs()[0].shape
    logger.info(f"Input shape: {input_shape}")
    
    # get output shape
    output_shape = sess.get_outputs()[0].shape
    logger.info(f"Output shape: {output_shape}")
    
    # transform data to the expected tensor type
    test_data_transformed = test_data_transformed.astype(np.float32)

    # Evaluate model
    results = sess.run(None, {sess.get_inputs()[0].name: test_data_transformed})
    
    # get predictions
    predictions = results[0]
    
    
    if ml_task == "binary_classification":
        # convert predictions to labels
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
        
        # save to datframe
        df_evaluation = pd.DataFrame([[experiment, device, accuracy, f1, auc, recall, precision, balanced_accuracy, matthews]], columns=["experiment", "device", "accuracy", "f1", "auc", "recall", "precision", "balanced_accuracy", "matthews"])
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
            
        # save to datframe
        df_evaluation = pd.DataFrame([[experiment, device, mae, mse, mape, smape]], columns=["experiment", "device", "mae", "mse", "mape", "smape"])
    
    with open("poc_energy_efficiency_crypto/crypto_spider_5g_fcnn_optimized_benchmark_evaluation.pkl", "wb") as f:
        pickle.dump(test_results, f)
    
    return df_evaluation

def perform_evaluation_tflite(experiment, device="CPU"):
    assert device in ["CPU", "GPU"]

    logger.info(f"Evaluating {experiment} model on test set")

    # Load model
    interpreter = load_model(ext="tflite")
    interpreter.allocate_tensors()

    # Load test set
    test_data_transformed = standard.transform(df_test[gf])
    
    # get input shape
    input_shape = interpreter.get_input_details()[0]["shape"]
    logger.info(f"Input shape: {input_shape}")
    
    # get output shape
    output_shape = interpreter.get_output_details()[0]["shape"]
    logger.info(f"Output shape: {output_shape}")
    
    # transform data to the expected tensor type
    input_details = interpreter.get_input_details()[0]
    dtype = input_details["dtype"]    
    input_data = np.array(test_data_transformed, dtype=dtype)
    
    # reshape model input
    batch_size = 256
    
    input_details = interpreter.get_input_details()[0]
    interpreter.resize_tensor_input(input_details['index'], (batch_size, input_data.shape[1]))
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    
    preds = []

    # create batches of test_data_transformed
    for i in range(0, len(test_data_transformed), batch_size):
        batch = test_data_transformed[i:i+batch_size]
        # print(f"Batch {i//batch_size} has {len(batch)} elements")
        
        batch_data = np.array(batch, dtype=dtype)
        
        if len(batch) == batch_size:        
            interpreter.set_tensor(input_details['index'], batch_data)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details['index'])
            
            preds.append(output_data)
    
    predictions = np.concatenate(preds)
    
    # take the labels of the test set
    test_labels = df_test["tag"].values[:len(predictions)]

    # Evaluate model
    # accuracy = accuracy_score(test_labels, predictions)
    # balanced_accuracy = balanced_accuracy_score(test_labels, predictions)
    # f1 = f1_score(test_labels, predictions, average="weighted")

    # logger.info(f"Accuracy: {accuracy}")
    # logger.info(f"Balanced accuracy: {balanced_accuracy}")
    # logger.info(f"F1 score: {f1}")
    
    # conf_matrix = confusion_matrix(test_labels, predictions)
    # logger.info(f"Confusion matrix: {conf_matrix}")
    
    # test_results = {
    #     "accuracy": accuracy,
    #     "balanced_accuracy": balanced_accuracy,
    #     "f1": f1,
    # }

    # with open("poc_energy_efficiency_crypto/crypto_spider_5g_fcnn_optimized_benchmark_evaluation.pkl", "wb") as f:
    #     pickle.dump(test_results, f)
    
    # # save to datframe
    # df_evaluation = pd.DataFrame([[experiment, device, accuracy, balanced_accuracy, f1]], columns=["experiment", "device", "accuracy", "balanced_accuracy", "f1"])
    
    if ml_task == "binary_classification":
        # convert predictions to labels
        predictions = np.argmax(predictions, axis=1)
        
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
        
        # save to datframe
        df_evaluation = pd.DataFrame([[experiment, device, accuracy, f1, auc, recall, precision, balanced_accuracy, matthews]], columns=["experiment", "device", "accuracy", "f1", "auc", "recall", "precision", "balanced_accuracy", "matthews"])
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
            
        # save to datframe
        df_evaluation = pd.DataFrame([[experiment, device, mae, mse, mape, smape]], columns=["experiment", "device", "mae", "mse", "mape", "smape"])
    
    with open("poc_energy_efficiency_crypto/crypto_spider_5g_fcnn_optimized_benchmark_evaluation.pkl", "wb") as f:
        pickle.dump(test_results, f)
    
    return df_evaluation