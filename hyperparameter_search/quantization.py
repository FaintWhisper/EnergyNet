# #### Quantization

# In[ ]:


from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV
import tensorflow as tf
import tensorflow_model_optimization as tfmot
import numpy as np
import pandas as pd


class QuantizeConfig(tfmot.quantization.keras.QuantizeConfig):
    def __init__(self, quantization_config = "LastValueQuantizer", num_bits=8, symmetric=True, narrow_range=False, per_axis=False):
        self.quantization_config = quantization_config
        self.num_bits = num_bits
        self.symmetric = symmetric
        self.narrow_range = narrow_range
        self.per_axis = per_axis
        
        if self.quantization_config == "LastValueQuantizer":
            self.quantizer = tfmot.quantization.keras.quantizers.LastValueQuantizer(num_bits=self.num_bits, symmetric=self.symmetric, narrow_range=self.narrow_range, per_axis=self.per_axis)
        elif self.quantization_config == "MovingAverageQuantizer":
            self.quantizer = tfmot.quantization.keras.quantizers.MovingAverageQuantizer(num_bits=self.num_bits, symmetric=self.symmetric, narrow_range=self.narrow_range, per_axis=self.per_axis)
        elif self.quantization_config == "AllValuesQuantizer":
            self.quantizer = tfmot.quantization.keras.quantizers.AllValuesQuantizer(num_bits=self.num_bits, symmetric=self.symmetric, narrow_range=self.narrow_range, per_axis=self.per_axis)

    # Configure how to quantize weights.
    def get_weights_and_quantizers(self, layer):
        return [
            (
                layer.kernel,
                self.quantizer,
            )
        ]

    # Configure how to quantize activations.
    def get_activations_and_quantizers(self, layer):
        return [
            (
                layer.kernel,
                self.quantizer,
            )
        ]

    def set_quantize_weights(self, layer, quantize_weights):
        # Add this line for each item returned in `get_weights_and_quantizers`
        # , in the same order
        layer.kernel = quantize_weights[0]

    def set_quantize_activations(self, layer, quantize_activations):
        # Add this line for each item returned in `get_activations_and_quantizers`
        # , in the same order.
        layer.activation = quantize_activations[0]

    # Configure how to quantize outputs (may be equivalent to activations).
    def get_output_quantizers(self, layer):
        return []

    def get_config(self):
        # serialize the quantizer
        return {
            "quantization_config": self.quantization_config,
            "num_bits": self.num_bits,
            "symmetric": self.symmetric,
            "narrow_range": self.narrow_range,
            "per_axis": self.per_axis,
        }


class QuantizationEstimator(BaseEstimator):
    def __init__(
        self, batch_size=256, epochs=10, quantization_config="LastValueQuantizer"
    ):
        self.batch_size = batch_size
        self.epochs = epochs
        self.quantization_config = quantization_config

    def fit(self, X, y):
        # Fine-tune pretrained model with quantization aware training

        # Define the quantization configuration
        quantize_config = QuantizeConfig(self.quantization_config)
        
        model = tf.keras.Sequential()
        
        # annotate baseline model
        for layer in baseline.layers:
            annotated_layer = tfmot.quantization.keras.quantize_annotate_layer(layer, quantize_config)
            model.add(annotated_layer)
        
        model = tfmot.quantization.keras.quantize_annotate_model(model)

        # Apply quantization to the model
        with tfmot.quantization.keras.quantize_scope({'DefaultDenseQuantizeConfig': QuantizeConfig}):
            q_aware_model = tfmot.quantization.keras.quantize_apply(model)
            
        q_aware_model.compile(
            optimizer="adam",
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
            metrics=["accuracy"],
        )

        validation_split = 0.1

        history = q_aware_model.fit(
            X,
            y,
            validation_split=validation_split,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=0,
        )

        converter = tf.lite.TFLiteConverter.from_keras_model(q_aware_model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        quantized_model = converter.convert()

        # Save the model to disk
        self.quantized_tflite_model_file = "model.tflite"

        with open(self.quantized_tflite_model_file, "wb") as f:
            f.write(quantized_model)
            
        return self

    def score(self, X, y):
        # Load model
        interpreter = tf.lite.Interpreter(model_path=self.quantized_tflite_model_file)
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
        input_data = np.array(test_data_transformed, dtype=dtype)

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
            test_labels, np.argmax(predictions, axis=1)
        )

        return balanced_accuracy


# In[ ]:


with tf.device('/device:CPU:0'):
    param_grid = {'batch_size': [256, 512, 1024, 2048], 'epochs': [10, 20, 30], 'quantization_config': ["LastValueQuantizer", "MovingAverageQuantizer", "AllValuesQuantizer",]}

    grid_search = GridSearchCV(estimator=QuantizationEstimator(), param_grid=param_grid, cv=3, verbose=2)
    grid_search.fit(data_transformed, pd.get_dummies(df_train["tag"]))

    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best score: {grid_search.best_score_}")


# In[46]:


from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV
import tensorflow as tf
import tensorflow_model_optimization as tfmot
import numpy as np
import pandas as pd


class QuantizeConfig(tfmot.quantization.keras.QuantizeConfig):
    def __init__(self, quantization_config = "LastValueQuantizer", num_bits=8, symmetric=True, narrow_range=False, per_axis=False):
        self.quantization_config = quantization_config
        self.num_bits = num_bits
        self.symmetric = symmetric
        self.narrow_range = narrow_range
        self.per_axis = per_axis
        
        if self.quantization_config == "LastValueQuantizer":
            self.quantizer = tfmot.quantization.keras.quantizers.LastValueQuantizer(num_bits=self.num_bits, symmetric=self.symmetric, narrow_range=self.narrow_range, per_axis=self.per_axis)
        elif self.quantization_config == "MovingAverageQuantizer":
            self.quantizer = tfmot.quantization.keras.quantizers.MovingAverageQuantizer(num_bits=self.num_bits, symmetric=self.symmetric, narrow_range=self.narrow_range, per_axis=self.per_axis)
        elif self.quantization_config == "AllValuesQuantizer":
            self.quantizer = tfmot.quantization.keras.quantizers.AllValuesQuantizer(num_bits=self.num_bits, symmetric=self.symmetric, narrow_range=self.narrow_range, per_axis=self.per_axis)

    # Configure how to quantize weights.
    def get_weights_and_quantizers(self, layer):
        return [
            (
                layer.kernel,
                self.quantizer,
            )
        ]

    # Configure how to quantize activations.
    def get_activations_and_quantizers(self, layer):
        return [
            (
                layer.kernel,
                self.quantizer,
            )
        ]

    def set_quantize_weights(self, layer, quantize_weights):
        # Add this line for each item returned in `get_weights_and_quantizers`
        # , in the same order
        layer.kernel = quantize_weights[0]

    def set_quantize_activations(self, layer, quantize_activations):
        # Add this line for each item returned in `get_activations_and_quantizers`
        # , in the same order.
        layer.activation = quantize_activations[0]

    # Configure how to quantize outputs (may be equivalent to activations).
    def get_output_quantizers(self, layer):
        return []

    def get_config(self):
        # serialize the quantizer
        return {
            "quantization_config": self.quantization_config,
            "num_bits": self.num_bits,
            "symmetric": self.symmetric,
            "narrow_range": self.narrow_range,
            "per_axis": self.per_axis,
        }


class QuantizationEstimator(BaseEstimator):
    def __init__(
        self, batch_size=256, epochs=10,
    ):
        self.batch_size = batch_size
        self.epochs = epochs

    def fit(self, X, y):
        # Fine-tune pretrained model with quantization aware training

        q_aware_model = tfmot.quantization.keras.quantize_model(baseline)
        q_aware_model.compile(
            optimizer="adam", loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False), metrics=["accuracy"]
        )
        
        validation_split = 0.1

        history = q_aware_model.fit(X, y, validation_split=validation_split, epochs=self.epochs, batch_size=self.batch_size)

        converter = tf.lite.TFLiteConverter.from_keras_model(q_aware_model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        quantized_model = converter.convert()

        # Save the model to disk
        self.quantized_tflite_model_file = "model.tflite"

        with open(self.quantized_tflite_model_file, "wb") as f:
            f.write(quantized_model)
            
        return self

    def score(self, X, y):
        # Load model
        interpreter = tf.lite.Interpreter(model_path=self.quantized_tflite_model_file)
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

        return balanced_accuracy

    def evaluate(self, X, y):
        # Load model
        interpreter = tf.lite.Interpreter(model_path=self.quantized_tflite_model_file)
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


# In[45]:


with tf.device('/device:CPU:0'):
    param_grid = {'batch_size': [256, 512, 1024, 2048], 'epochs': [10, 20, 30]}

    grid_search = GridSearchCV(estimator=QuantizationEstimator(), param_grid=param_grid, cv=3, verbose=2)
    grid_search.fit(data_transformed, pd.get_dummies(df_train["tag"]))

    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best score: {grid_search.best_score_}")


# In[54]:


qe = QuantizationEstimator(batch_size=256, epochs=10)
qe.fit(data_transformed, pd.get_dummies(df_train["tag"]))
qe.evaluate(test_data_transformed, pd.get_dummies(df_test["tag"]))

