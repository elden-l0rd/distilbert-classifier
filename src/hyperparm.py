from tqdm import tqdm
import itertools
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import StratifiedKFold

def hyperparameter_tuning(X_train_tfidf, y_train):
    num_epochs = 100
    num_classes = 5
    vocab_size = X_train_tfidf.shape[1]
    dropout_rates = [0.2, 0.3, 0.4, 0.5]
    activations_list = ['relu', 'sigmoid', 'elu', 'tanh']
    num_neurons = [32, 64, 128, 256]
    opt_lr = [1e-2, 1e-3, 1e-4]
    L2_lr = [1e-2, 1e-3, 1e-4]
    best_loss = float('inf')
    best_params = None

    hyperparam_combi = itertools.product(dropout_rates, num_neurons, activations_list, opt_lr, L2_lr)

    for dr, nn, al, olr, l2lr in tqdm(hyperparam_combi):
        modelTest = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(vocab_size,)),
            tf.keras.layers.Dense(nn*2, activation=al),
            tf.keras.layers.Dropout(dr),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(nn, activation=al),
            tf.keras.layers.Dropout(dr),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(nn//2, activation=al),
            tf.keras.layers.Dense(num_classes, kernel_regularizer=tf.keras.regularizers.L2(l2=1e-2), activation='softmax')
        ])

        optimizer = tf.keras.optimizers.Adam(olr)
        modelTest.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy")
        
        early_stop = EarlyStopping(
            monitor="loss",
            patience=5,
            verbose=0,
            restore_best_weights=True
        )

        histTest = modelTest.fit(
            X_train_tfidf, y_train,
            batch_size=16,
            epochs=num_epochs,
            verbose=0,
            callbacks=[early_stop]
        )

        min_loss = min(histTest.history['loss'])
        if min_loss < best_loss:
            best_loss = min_loss
            best_params = (dr, nn, al, olr, l2lr)
            
    print(f"Final Best Hyperparameters: \nDropout: {best_params[0]},\nActivation: {best_params[2]},\nHidden Units: {best_params[1]},\nL2 Reg: {best_params[4]},\nLR: {best_params[3]},\nBest Loss: {best_loss}")
    return

def Kfold_tuning(df, X, y, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    dropout_rates = [0.2, 0.3, 0.4, 0.5]
    activations_list = ['relu', 'leaky_relu', 'elu', 'tanh']
    num_neurons = [32, 64, 128, 256]
    opt_lr = [1e-2, 1e-3, 1e-4]
    L2_lr = [1e-2, 1e-3, 1e-4]
    
    best_val_acc = 0
    best_params = None
    
    hyperparam_combi = itertools.product(dropout_rates, num_neurons, activations_list, opt_lr, L2_lr)
    
    for dr, nn, act, lr, l2 in hyperparam_combi:
        fold_results = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(df.index, df['label'])):
            print(f"\nTraining on Fold {fold+1} with params: dr={dr}, nn={nn}, act={act}, lr={lr}, l2={l2}")
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]
            
            model = build_model(nn, act, dr, lr, l2, X.shape[1], len(set(y)))
            
            early_stop = EarlyStopping(
                monitor="val_loss",
                patience=5,
                verbose=0,
                restore_best_weights=True
            )
            history = model.fit(
                X_train_fold, y_train_fold,
                batch_size=16,
                epochs=100,
                validation_data=(X_val_fold, y_val_fold),
                callbacks=[early_stop],
                verbose=0
            )
            
            val_acc = max(history.history['val_accuracy'])
            fold_results.append(val_acc)
        
        avg_val_acc = sum(fold_results) / len(fold_results)
        print(f"Average Validation Accuracy for current params: {avg_val_acc}")
        
        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            best_params = (dr, nn, act, lr, l2)
    
    print(f"Best Validation Accuracy: {best_val_acc} with parameters: {best_params}")
    print(f"Dropout Rate: {best_params[0]}")
    print(f"Number of Neurons: {best_params[1]}")
    print(f"Activation Function: {best_params[2]}")
    print(f"Learning Rate: {best_params[3]}")
    print(f"L2 Regularization: {best_params[4]}")

    return

def build_model(nn, act, dr, lr, l2, input_dim, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(nn*2, activation=act),
        tf.keras.layers.Dropout(dr),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(nn, activation=act),
        tf.keras.layers.Dropout(dr),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(nn//2, activation=act),
        tf.keras.layers.Dense(num_classes, kernel_regularizer=tf.keras.regularizers.L2(l2=l2), activation='softmax')
    ])
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=['accuracy'])
    return model
