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
