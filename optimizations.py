import tensorflow as tf

def full_integer_quantization(trained_model):
    def dataset_generator():
        for data in tf.data.Dataset.from_tensor_slices((data_transformed)).batch(1).take(int(len(data_transformed) * 0.25)):
            yield [tf.dtypes.cast(data, tf.float32)]

    converter = tf.lite.TFLiteConverter.from_keras_model(trained_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = dataset_generator
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8  # or tf.uint8
    converter.inference_output_type = tf.int8  # or tf.uint8
    quantized_model = converter.convert()

    logger.info("Applied Full-integer Quantization")

    # Save the model to disk
    quantized_tflite_model_file = 'model.tflite'
    
    with open(quantized_tflite_model_file, 'wb') as f:
        f.write(quantized_model)

    return quantized_tflite_model_file


def float16_quantization(trained_model):
    def dataset_generator():
        for data in tf.data.Dataset.from_tensor_slices((data_transformed)).batch(1).take(int(len(data_transformed) * 0.25)):
            yield [tf.dtypes.cast(data, tf.float32)]

    converter = tf.lite.TFLiteConverter.from_keras_model(trained_model)
    converter.representative_dataset = dataset_generator
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.float16]
    quantized_model = converter.convert()

    logger.info("Applied float16 Activations and int8 weights-Quantization")

    # Save the model to disk
    quantized_tflite_model_file = 'model.tflite'
    
    with open(quantized_tflite_model_file, 'wb') as f:
        f.write(quantized_model)

    return quantized_tflite_model_file


def float16_int8_quantization(trained_model):
    def dataset_generator():
        for data in tf.data.Dataset.from_tensor_slices((data_transformed)).batch(1).take(int(len(data_transformed) * 0.25)):
            yield [tf.dtypes.cast(data, tf.float32)]

    converter = tf.lite.TFLiteConverter.from_keras_model(trained_model)
    converter.representative_dataset = dataset_generator
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8]
    quantized_model = converter.convert()

    logger.info("Applied float16 Activations and int8 weights-Quantization")

    # Save the model to disk
    quantized_tflite_model_file = 'model.tflite'
    
    with open(quantized_tflite_model_file, 'wb') as f:
        f.write(quantized_model)

    return quantized_tflite_model_file

# Fine-tune pretrained model with pruning
def pruning(model, convert_to_tflite=True):
    prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

    # Compute end step to finish pruning after 10% of pre-training epochs
    batch_size = 256
    # n_epochs = 20
    n_epochs = 10
    
    validation_split = 0.1

    num_samples = len(df_train)
    end_step = np.ceil(num_samples / batch_size).astype(np.int32) * epochs

    pruning_params = {
        "pruning_schedule": tfmot.sparsity.keras.PolynomialDecay(
            initial_sparsity=0.50, final_sparsity=0.80, begin_step=0, end_step=end_step
        )
    }

    model_for_pruning = prune_low_magnitude(model, **pruning_params)

    model_for_pruning.compile(
        optimizer="adam", loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False), metrics=["accuracy"]
    )

    callbacks = [
        tfmot.sparsity.keras.UpdatePruningStep(),
    ]
    
    history = model_for_pruning.fit(data_transformed, pd.get_dummies(df_train['tag']), validation_split=validation_split, epochs=n_epochs, batch_size=batch_size, callbacks=callbacks)

    pruned_model = tfmot.sparsity.keras.strip_pruning(model_for_pruning)
    
    if convert_to_tflite:
        # convert to tflite
        converter = tf.lite.TFLiteConverter.from_keras_model(pruned_model)
        converter.optimizations = [tf.lite.Optimize.EXPERIMENTAL_SPARSITY, tf.lite.Optimize.DEFAULT]
        pruned_tflite_model = converter.convert()
        
        # Save the model to disk
        pruned_tflite_model_file = 'model.tflite'
        
        with open(pruned_tflite_model_file, 'wb') as f:
            f.write(pruned_tflite_model)

        return pruned_tflite_model_file, history
        
        # return pruned_model, history
    else:
        return pruned_model, history


# Quantization Aware Training
def quantization_aware_training(model):
    quantize_model = tfmot.quantization.keras.quantize_model

    q_aware_model = quantize_model(model)
    q_aware_model.compile(
        optimizer="adam", loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False), metrics=["accuracy"]
    )
    
    batch_size = 256
    n_epochs = 10
    validation_split = 0.1

    history = q_aware_model.fit(data_transformed, pd.get_dummies(df_train['tag']), validation_split=validation_split, epochs=n_epochs, batch_size=batch_size)

    converter = tf.lite.TFLiteConverter.from_keras_model(q_aware_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    quantized_model = converter.convert()
    
    # Save the model to disk
    quantized_tflite_model_file = 'model.tflite'
    
    with open(quantized_tflite_model_file, 'wb') as f:
        f.write(quantized_model)

    return quantized_tflite_model_file, history


# Knowledge Distillation
class Distiller(keras.Model):
    def __init__(self, student, teacher):
        super(Distiller, self).__init__()
        self.teacher = teacher
        self.student = student

    def compile(
        self, optimizer, metrics, student_loss_fn, distillation_loss_fn, alpha=0.1, temperature=3,
    ):
        """ Configure the distiller.

        Args:
            optimizer: Keras optimizer for the student weights
            metrics: Keras metrics for evaluation
            student_loss_fn: Loss function of difference between student
                predictions and ground-truth
            distillation_loss_fn: Loss function of difference between soft
                student predictions and soft teacher predictions
            alpha: weight to student_loss_fn and 1-alpha to distillation_loss_fn
            temperature: Temperature for softening probability distributions.
                Larger temperature gives softer distributions.
        """
        super(Distiller, self).compile(optimizer=optimizer, metrics=metrics)
        self.student_loss_fn = student_loss_fn
        self.distillation_loss_fn = distillation_loss_fn
        self.alpha = alpha
        self.temperature = temperature

    def train_step(self, data):
        # Unpack data
        x, y = data

        # Forward pass of teacher
        teacher_predictions = self.teacher(x, training=False)

        with tf.GradientTape() as tape:
            # Forward pass of student
            student_predictions = self.student(x, training=True)

            # Compute losses
            student_loss = self.student_loss_fn(y, student_predictions)

            # Compute scaled distillation loss from https://arxiv.org/abs/1503.02531
            # The magnitudes of the gradients produced by the soft targets scale
            # as 1/T^2, multiply them by T^2 when using both hard and soft targets.
            distillation_loss = (
                self.distillation_loss_fn(
                    tf.nn.softmax(teacher_predictions / self.temperature, axis=1),
                    tf.nn.softmax(student_predictions / self.temperature, axis=1),
                )
                * self.temperature ** 2
            )

            loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss

        # Compute gradients
        trainable_vars = self.student.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update the metrics configured in `compile()`.
        self.compiled_metrics.update_state(y, student_predictions)

        # Return a dict of performance
        results = {m.name: m.result() for m in self.metrics}
        results.update({"student_loss": student_loss, "distillation_loss": distillation_loss})

        return results

    def test_step(self, data):
        # Unpack the data
        x, y = data

        # Compute predictions
        y_prediction = self.student(x, training=False)

        # Calculate the loss
        student_loss = self.student_loss_fn(y, y_prediction)

        # Update the metrics.
        self.compiled_metrics.update_state(y, y_prediction)

        # Return a dict of performance
        results = {m.name: m.result() for m in self.metrics}
        results.update({"student_loss": student_loss})

        return results


def knowledge_distillation(teacher, convert_to_tflite=True):
    logger.info("Applying Knowledge Distillation")
    student = small_model(input_dim=len(gf), n_output=2)

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
    )
    
    if convert_to_tflite:
        # convert to tflite
        converter = tf.lite.TFLiteConverter.from_keras_model(distiller.student)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        distilled_tflite_model = converter.convert()
        
        # Save the model to disk
        distilled_tflite_model_file = 'model.tflite'
        
        with open(distilled_tflite_model_file, 'wb') as f:
            f.write(distilled_tflite_model)

        return distilled_tflite_model_file, history

        # return distiller.student, history
    else:
        return distiller.student, history