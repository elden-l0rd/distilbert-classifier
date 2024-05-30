'''
Defines a feedforawrd neural network model architecture
'''

import pickle
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.feature_extraction.text import TfidfVectorizer

def vectorize(df_train, df_test):
    tfidf_vectorizer = TfidfVectorizer()
    
    df_train['Desc'] = df_train['Desc'].apply(lambda x: ' '.join(x))
    df_test['Desc'] = df_test['Desc'].apply(lambda x: ' '.join(x))

    X_train_tfidf = tfidf_vectorizer.fit_transform(df_train['Desc']).toarray()
    X_test_tfidf = tfidf_vectorizer.transform(df_test['Desc']).toarray()

    y_train = df_train['label'].values
    y_test = df_test['label'].values

    return X_train_tfidf, X_test_tfidf, y_train, y_test

def initialise_model(hidden_units, num_classes, vocab_size, dropout, activation, lr, l2_reg):
    OPTIMIZER = tf.keras.optimizers.Adam(lr)

    model = tf.keras.Sequential([        
        tf.keras.layers.Input(shape=(vocab_size,)),
        tf.keras.layers.Dense(hidden_units*2, activation=activation),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(hidden_units, activation=activation),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(hidden_units//2, activation=activation),
        tf.keras.layers.Dense(num_classes, kernel_regularizer=tf.keras.regularizers.L2(l2=l2_reg), activation=activation),
    ])

    model.compile(optimizer=OPTIMIZER, loss="sparse_categorical_crossentropy", metrics=['accuracy'])
    model.summary()
    # plot_model(model, show_shapes=True, show_layer_names=True)
    return model

early_stop = EarlyStopping(
    monitor="loss",
    patience=5,
    verbose=1,
    restore_best_weights=True
)

def train_loop(model, X_train_tfidf, y_train, NUM_EPOCHS, BATCH_SIZE):
    hist = model.fit(
        X_train_tfidf, y_train,
        batch_size=BATCH_SIZE,
        epochs=NUM_EPOCHS,
        # validation_data=(X_val_tfidf, y_val),
        verbose=1,
        callbacks=[early_stop,]
    )
    return hist, model
