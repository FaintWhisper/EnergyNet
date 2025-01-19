# #### KD

# In[60]:


teacher = baseline_model(input_dim=len(gf), n_output=2)
teacher, history = train_model(teacher)


# In[123]:


from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV
import pickle

class DistillerEstimator(BaseEstimator):
    def __init__(self, alpha=0.1, temperature=3):
        self.alpha = alpha
        self.temperature = temperature
    
    def fit(self, X, y):
        student = small_model(input_dim=len(gf), n_output=2)
        distiller = Distiller(student=student, teacher=teacher)
        distiller.compile(
            optimizer=keras.optimizers.Adam(),
            metrics=[keras.metrics.CategoricalAccuracy()],
            student_loss_fn=keras.losses.CategoricalCrossentropy(from_logits=False),
            distillation_loss_fn=keras.losses.KLDivergence(),
            alpha=self.alpha,
            temperature=self.temperature,
        )
        es = EarlyStopping(
            monitor="val_student_loss",
            mode="min",
            patience=es_patience,
            restore_best_weights=es_restore_best_weights,
        )
        history = distiller.fit(
            X,
            y,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[es],
            verbose=False,
        )
        self.distiller_ = distiller
        return self
    
    # def score(self, X, y):
    #     test_data_transformed = standard.transform(df_test[gf])
    #     predictions = self.distiller_.student.predict(test_data_transformed)
    #     balanced_accuracy = balanced_accuracy_score(df_test["tag"], np.argmax(predictions, axis=1))
    #     return balanced_accuracy
    def score(self, X, y):
        predictions = self.distiller_.student.predict(X)
        balanced_accuracy = balanced_accuracy_score(np.argmax(y, axis=1), np.argmax(predictions, axis=1))
        
        return balanced_accuracy
    
    def evaluate(self, X, y):
        predictions = self.distiller_.student.predict(X)
        print(f"predictions shape: {predictions.shape}")
        y = np.array(y)
        accuracy = accuracy_score(np.argmax(y, axis=1), np.argmax(predictions, axis=1))
        balanced_accuracy = balanced_accuracy_score(np.argmax(y, axis=1), np.argmax(predictions, axis=1))
        f1 = f1_score(np.argmax(y, axis=1), np.argmax(predictions, axis=1), average="weighted")
        
        accuracy = round(accuracy, 3)
        balanced_accuracy = round(balanced_accuracy, 3)
        f1 = round(f1, 3)
        
        print(f"Accuracy: {accuracy}")
        print(f"Balanced accuracy: {balanced_accuracy}")
        print(f"F1 score: {f1}")


# In[124]:


param_grid = {'alpha': [0.1, 0.2, 0.3, 0.4, 0.5],
              'temperature': [0.5, 1, 1.5, 2, 2.5, 3]}

grid_search = GridSearchCV(estimator=DistillerEstimator(), param_grid=param_grid, cv=3, verbose=2)
grid_search.fit(data_transformed, pd.get_dummies(df_train["tag"]))

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_}")


# In[125]:


de = DistillerEstimator(alpha=0.1, temperature=2)
de.fit(data_transformed, pd.get_dummies(df_train["tag"]))
de.evaluate(test_data_transformed, pd.get_dummies(df_test["tag"]))


