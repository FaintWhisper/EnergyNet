def train_model(model, verbose=False, extra_callbacks=[]):
    print("Training model")
    print("Verbose: {}".format(verbose))
    print("Extra callbacks: {}".format(extra_callbacks))
    
    # es = EarlyStopping(monitor='val_loss', mode='min', patience=es_patience, verbose=verbose, restore_best_weights=es_restore_best_weights)
    # history = model.fit(data_transformed, pd.get_dummies(df_train['tag']), validation_split=validation_split, epochs=epochs, batch_size=4096, callbacks=[es] + extra_callbacks, verbose=verbose)
    # history = model.fit(data_transformed, pd.get_dummies(df_train['tag']), validation_split=validation_split, epochs=epochs, batch_size=4096, callbacks=extra_callbacks, verbose=verbose)
    history = model.fit(data_transformed, pd.get_dummies(df_train['tag']), validation_split=validation_split, epochs=50, batch_size=4096, callbacks=extra_callbacks, verbose=verbose)

    return model, history