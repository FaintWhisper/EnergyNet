# #### PT Quantization

# In[35]:


baseline = baseline_model(input_dim=len(gf), n_output=2)
baseline, history = train_model(baseline)


# In[36]:


def dataset_generator():
    dataset_size = 0.1
    print("Generating dataset for float16 activations and int8 weights quantization")
    print("Original Dataset size: ", int(len(data_transformed)))
    print("Dataset size: ", int(len(data_transformed) * dataset_size))
    
    for data in tf.data.Dataset.from_tensor_slices((data_transformed)).batch(1).take(int(len(data_transformed) * dataset_size)):
        yield [tf.dtypes.cast(data, tf.float32)]

converter = tf.lite.TFLiteConverter.from_keras_model(baseline)
converter.representative_dataset = dataset_generator
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.float16]
quantized_model = converter.convert()


# In[39]:


def float16_quantization(trained_model, dataset_size: float = 0.25):
    def dataset_generator(): # Does not need a representative dataset
        print("Generating dataset for float16 weights quantization")
        print("Original Dataset size: ", int(len(data_transformed)))
        print("Dataset size: ", int(len(data_transformed) * dataset_size))
        
        for data in tf.data.Dataset.from_tensor_slices((data_transformed)).batch(1).take(int(len(data_transformed) * dataset_size)):
            yield [tf.dtypes.cast(data, tf.float32)]

    converter = tf.lite.TFLiteConverter.from_keras_model(trained_model)
    # converter.representative_dataset = dataset_generator
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.float16]
    quantized_model = converter.convert()

    logger.info("Applied float16 weights quantization")

    # Save the model to disk
    quantized_tflite_model_file = 'model.tflite'
    
    with open(quantized_tflite_model_file, 'wb') as f:
        f.write(quantized_model)

    return quantized_tflite_model_file

dataset_sizes = [0.1, 0.25, 0.5, 0.75, 1.0]

for dataset_size in dataset_sizes:
    tflite_model = float16_quantization(baseline, dataset_size)

    # evaluate
    interpreter = tf.lite.Interpreter(model_path=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    test_data_transformed = standard.transform(df_test[gf])
    predictions = []

    # cast data to float32
    test_data_transformed = test_data_transformed.astype(np.float32)

    for i in range(len(test_data_transformed)):
        interpreter.set_tensor(input_details[0]['index'], test_data_transformed[i].reshape(1, -1))
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        predictions.append(np.argmax(output_data))
    
    accuracy = accuracy_score(df_test["tag"], predictions)
    balanced_accuracy = balanced_accuracy_score(df_test["tag"], predictions)
    f1 = f1_score(df_test["tag"], predictions, average="weighted")
    
    accuracy = round(accuracy, 4)
    balanced_accuracy = round(balanced_accuracy, 4)
    f1 = round(f1, 4)
    
    print(f"Accuracy: {accuracy}")
    print(f"Balanced accuracy: {balanced_accuracy}")
    print(f"F1 score: {f1}")


# In[40]:


def int8_quantization(trained_model, dataset_size: float = 0.25):
    def dataset_generator():
        print("Generating dataset for int8 weights quantization")
        print("Original Dataset size: ", int(len(data_transformed)))
        print("Dataset size: ", int(len(data_transformed) * dataset_size))
        
        for data in tf.data.Dataset.from_tensor_slices((data_transformed)).batch(1).take(int(len(data_transformed) * dataset_size)):
            yield [tf.dtypes.cast(data, tf.float32)]

    converter = tf.lite.TFLiteConverter.from_keras_model(trained_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = dataset_generator
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8  # or tf.uint8
    converter.inference_output_type = tf.int8  # or tf.uint8
    quantized_model = converter.convert()

    logger.info("Applied int8 weights quantization")

    # Save the model to disk
    quantized_tflite_model_file = 'model.tflite'
    
    with open(quantized_tflite_model_file, 'wb') as f:
        f.write(quantized_model)

    return quantized_tflite_model_file

dataset_sizes = [0.1, 0.25, 0.5, 0.75, 1.0]

for dataset_size in dataset_sizes:
    tflite_model = int8_quantization(baseline, dataset_size)

    # evaluate
    interpreter = tf.lite.Interpreter(model_path=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    test_data_transformed = standard.transform(df_test[gf])
    predictions = []

    # cast data to float32
    test_data_transformed = test_data_transformed.astype(np.int8)

    for i in range(len(test_data_transformed)):
        interpreter.set_tensor(input_details[0]['index'], test_data_transformed[i].reshape(1, -1))
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        predictions.append(np.argmax(output_data))
        
    accuracy = accuracy_score(df_test["tag"], predictions)
    balanced_accuracy = balanced_accuracy_score(df_test["tag"], predictions)
    f1 = f1_score(df_test["tag"], predictions, average="weighted")
    
    accuracy = round(accuracy, 4)
    balanced_accuracy = round(balanced_accuracy, 4)
    f1 = round(f1, 4)
    
    print(f"Accuracy: {accuracy}")
    print(f"Balanced accuracy: {balanced_accuracy}")
    print(f"F1 score: {f1}")


# In[41]:


def float16_activations_int8_weights_quantization(trained_model, dataset_size: float = 0.25):
    def dataset_generator():
        print("Generating dataset for float16 activations and int8 weights quantization")
        print("Original Dataset size: ", int(len(data_transformed)))
        print("Dataset size: ", int(len(data_transformed) * dataset_size))
        
        for data in tf.data.Dataset.from_tensor_slices((data_transformed)).batch(1).take(int(len(data_transformed) * dataset_size)):
            yield [tf.dtypes.cast(data, tf.float32)]

    converter = tf.lite.TFLiteConverter.from_keras_model(trained_model)
    converter.representative_dataset = dataset_generator
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8]
    quantized_model = converter.convert()

    logger.info("Applied float16 activations and int8 weights quantization")

    # Save the model to disk
    quantized_tflite_model_file = 'model.tflite'
    
    with open(quantized_tflite_model_file, 'wb') as f:
        f.write(quantized_model)

    return quantized_tflite_model_file

dataset_sizes = [0.1, 0.25, 0.5, 0.75, 1.0]

for dataset_size in dataset_sizes:
    tflite_model = float16_activations_int8_weights_quantization(baseline, dataset_size)

    # evaluate
    interpreter = tf.lite.Interpreter(model_path=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    test_data_transformed = standard.transform(df_test[gf])
    predictions = []

    # cast data to float32
    test_data_transformed = test_data_transformed.astype(np.float32)

    for i in range(len(test_data_transformed)):
        interpreter.set_tensor(input_details[0]['index'], test_data_transformed[i].reshape(1, -1))
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        predictions.append(np.argmax(output_data))
        
    accuracy = accuracy_score(df_test["tag"], predictions)
    balanced_accuracy = balanced_accuracy_score(df_test["tag"], predictions)
    f1 = f1_score(df_test["tag"], predictions, average="weighted")
    
    accuracy = round(accuracy, 4)
    balanced_accuracy = round(balanced_accuracy, 4)
    f1 = round(f1, 4)
    
    print(f"Accuracy: {accuracy}")
    print(f"Balanced accuracy: {balanced_accuracy}")
    print(f"F1 score: {f1}")


# In[42]:


def dynamic_range_quantization(trained_model, dataset_size: float = 0.25): # Not implemented in the framework
    def dataset_generator(): # Does not need a representative dataset
        print("Generating dataset for dynamic range quantization")
        print("Original Dataset size: ", int(len(data_transformed)))
        print("Dataset size: ", int(len(data_transformed) * dataset_size))
        
        for data in tf.data.Dataset.from_tensor_slices((data_transformed)).batch(1).take(int(len(data_transformed) * dataset_size)):
            yield [tf.dtypes.cast(data, tf.float32)]

    converter = tf.lite.TFLiteConverter.from_keras_model(trained_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    quantized_model = converter.convert()

    logger.info("Applied dynamic range quantization")

    # Save the model to disk
    quantized_tflite_model_file = 'model.tflite'
    
    with open(quantized_tflite_model_file, 'wb') as f:
        f.write(quantized_model)

    return quantized_tflite_model_file

dataset_sizes = [0.1, 0.25, 0.5, 0.75, 1.0]

for dataset_size in dataset_sizes:
    tflite_model = dynamic_range_quantization(baseline, dataset_size)

    # evaluate
    interpreter = tf.lite.Interpreter(model_path=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    test_data_transformed = standard.transform(df_test[gf])
    predictions = []

    # cast data to float32
    test_data_transformed = test_data_transformed.astype(np.float32)

    for i in range(len(test_data_transformed)):
        interpreter.set_tensor(input_details[0]['index'], test_data_transformed[i].reshape(1, -1))
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        predictions.append(np.argmax(output_data))
        
    accuracy = accuracy_score(df_test["tag"], predictions)
    balanced_accuracy = balanced_accuracy_score(df_test["tag"], predictions)
    f1 = f1_score(df_test["tag"], predictions, average="weighted")
    
    accuracy = round(accuracy, 4)
    balanced_accuracy = round(balanced_accuracy, 4)
    f1 = round(f1, 4)
    
    print(f"Accuracy: {accuracy}")
    print(f"Balanced accuracy: {balanced_accuracy}")
    print(f"F1 score: {f1}")


