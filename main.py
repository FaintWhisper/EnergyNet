import gc
import glob
import itertools
import json
import math
import multiprocessing
import os
import re
import socket
import string
import subprocess
import time
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed, load
from tqdm import tqdm_notebook as tqdm

pd.set_option('display.max_columns', None)

import sys

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, matthews_corrcoef,
                             mean_absolute_error,
                             mean_absolute_percentage_error,
                             mean_squared_error, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

sys.path.append("..")

import copy
import gzip
import logging
import os
import pickle
import tempfile
import zipfile
from contextlib import contextmanager
from multiprocessing import Process
from pathlib import Path

import onnx
import onnxruntime as rt
import psutil
import seaborn as sns
import tensorflow as tf
import tensorflow_model_optimization as tfmot
import tf2onnx
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential, load_model


class EnergyNet:
    def __init__(self):
        # Constants
        self.idle_gpu_memory = os.getenv("IDLE_GPU_MEMORY", 4)
        self.log_file = 'logs/log.txt'
        self.logger = None    

    def check_gpu_available(self,):
        """
        Check if GPU is available.
        """
        return tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None)

    def check_gpu_is_idle(self,):
        """
        Check if GPU is idle.
        """

        is_gpu_idle = False
        
        # check that the GPU utilization is 0
        gpu_utilization = os.system("nvidia-smi --query-gpu=utilization.gpu --format=csv")
        
        if gpu_utilization != 0:
            print("GPU utilization is not 0")
            is_gpu_idle = False
            
        # check that GPU memory is completely free
        gpu_memory = os.system("nvidia-smi --query-gpu=memory.free --format=csv")
        
        if gpu_memory != 0:
            print("GPU memory is not completely free")
            is_gpu_idle = False
            
        return is_gpu_idle

    def set_gpu_memory_growth(self,):
        print("Setting GPU memory growth")
        
        physical_devices = tf.config.list_physical_devices('GPU')

        for physical_device in physical_devices:
            try:
                tf.config.experimental.set_memory_growth(physical_device, True)
            except RuntimeError as e:
                print(e)
                
    def set_logger(self,):
        # delete previous log file
        if os.path.exists(self.log_file):
            os.remove(self.log_file)
        
        # create logger
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        
        Path(os.path.dirname(self.log_file)).mkdir(parents=True, exist_ok=True)
        
        filehandler = logging.FileHandler(self.log_file)
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        filehandler.setFormatter(formatter)
        self.logger.addHandler(filehandler)

        # add stream handler
        streamhandler = logging.StreamHandler()
        streamhandler.setFormatter(formatter)
        self.logger.addHandler(streamhandler)
        
    def set_default_benchmarking_params(self,):
        self.num_trials = 5
        self.stats_sampling_rate = 1

        self.es_patience = 20
        self.es_restore_best_weights = True

        self.batch_size = 1024 # TODO: this should be an array and the benchmarking should be done for all values
        self.validation_split = 0.2
        self.epochs = 10000

        self.devices = ["CPU"]

        self.ml_task = "binary_classification" # "binary_classification" or "regression"

        # self.general_optimizations = ["runtime", "graph"] NOT USED

    def set_benchmarking_params(self,):
        # check if config file exists
        
        if not os.path.exists("config.json"):
            self.set_default_benchmarking_params()
        else:
            with open("config.json", "r") as f:
                config = json.load(f)

            self.num_trials = config["num_trials"]
            self.stats_sampling_rate = config["stats_sampling_rate"]

            self.es_patience = config["es_patience"]
            self.es_restore_best_weights = config["es_restore_best_weights"]

            self.batch_size = config["batch_size"]
            self.validation_split = config["validation_split"]
            self.epochs = config["epochs"]

            self.devices = config["devices"]

            self.ml_task = config["ml_task"]

            # self.general_optimizations = config["general_optimizations"] NOT USED
    
    def check_benchmarking_params(self,):
        # Check if the parameters are valid
        assert self.devices in [["CPU"], ["GPU"], ["CPU", "GPU"]], "devices must be either ['CPU'], ['GPU'] or ['CPU', 'GPU']"
        assert self.ml_task in ["binary_classification", "regression"], "ml_task must be either 'binary_classification' or 'regression'"

        assert self.num_trials > 0, "num_trials must be greater than 0"
        assert self.stats_sampling_rate > 0, "stats_sampling_rate must be greater than 0"
        assert self.batch_size > 0, "batch_size must be greater than 0"
        assert self.validation_split > 0 and self.validation_split < 1, "validation_split must be greater than 0 and less than 1"
        assert self.epochs > 0, "epochs must be greater than 0"
        assert self.es_patience > 0, "es_patience must be greater than 0"
        assert self.es_restore_best_weights in [True, False], "es_restore_best_weights must be either True or False"

        self.logger.info(f'Number of trials set to: {self.num_trials}')
        self.logger.info(f'Epochs set to: {self.epochs}')
        self.logger.info(f'Batch size set to: {self.batch_size}')
        self.logger.info(f'Validation split set to: {self.validation_split}')
        self.logger.info(f'Patience set to: {self.es_patience}')
        self.logger.info(f'Restore best weights set to: {self.es_restore_best_weights}')
        
    def get_train_and_test_data_paths(self,):
        """
        Get the paths of the train and test data.
        """
        
        # check if config file exists
        if not os.path.exists("config.json"):
            raise Exception("config.json not found")
        else:
            with open("config.json", "r") as f:
                config = json.load(f)
                
        train_data_path = config["train_data_path"]
        test_data_path = config["test_data_path"]
        
        return train_data_path, test_data_path

    def check_train_and_test_data(self, train_data_path: str, test_data_path: str):
        """
        Check if the train and test data are valid.
        """
        
        self.df_train = pd.read_csv(train_data_path)
        self.df_test = pd.read_csv(test_data_path)
        
        assert len(self.df_train) > 0, "df_train must contain at least one row"
        assert len(self.df_test) > 0, "df_test must contain at least one row"
        assert len(self.df_train.columns) > 0, "df_train must contain at least one column"
        assert len(self.df_test.columns) > 0, "df_test must contain at least one column"
        assert len(self.df_train.columns) == len(self.df_test.columns), "df_train and df_test must have the same number of columns"
        
        return self.df_train, self.df_test
    
    def get_experiments(self,):
        # read experiments from experiments YAML file
        experiments = yaml.load(open("experiments.yaml", "r"), Loader=yaml.FullLoader)
        
        return experiments

    def run(self,):
        is_gpu_available = self.check_gpu_available()
        
        if is_gpu_available:
            is_gpu_idle = self.check_gpu_is_idle()
            
            if is_gpu_idle:
                print("GPU is available and idle")
            else:
                print("GPU is available but not idle. Please, ensure that GPU is idle before running the framework.")
                sys.exit(0)
        
        self.set_gpu_memory_growth()
        self.set_logger()
        
        train_data_path, test_data_path = self.get_train_and_test_data_paths()
        self.check_train_and_test_data(train_data_path, test_data_path)
        
        self.experiments = self.get_experiments()
        
        # automatic hyperparameter tuning
        # TODO
        
        # run benchmark
        # TODO: import submodules
        results.clean_files()
        benchmark.run_benchmark(experiments=self.experiments, devices=self.devices)
        
        # get results
        df_times, df_training_stats, df_inference_stats, df_load_stats, df_evaluation = results.get_results(aggregate_by_time=True)
        results.print_raw_results(df_times, df_training_stats, df_inference_stats, df_load_stats, df_evaluation)
        
        # export results 
        df_times, df_training_stats, df_inference_stats, df_load_stats, df_evaluation = results.get_results(aggregate_by_time=True)
        results.export_raw_results(df_times, df_training_stats, df_inference_stats, df_load_stats, df_evaluation)
        
        # prepare results
        results.prepare_results(self.batch_size)
        
        # print results
        results.print_results()
        
        # plot results
        results.plot_results()
        

if __name__ == "__main__":
    en = EnergyNet()
    en.run()