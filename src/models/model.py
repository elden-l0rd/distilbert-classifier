import pickle
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Conv1D, GlobalMaxPooling1D, Concatenate, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold


def vectorize(df_train, df_test):
    tfidf_vectorizer = TfidfVectorizer()
    
    df_train['Desc'] = df_train['Desc'].apply(lambda x: ' '.join(x))
    df_test['Desc'] = df_test['Desc'].apply(lambda x: ' '.join(x))
    # df_dev['Desc'] = df_dev['Desc'].apply(lambda x: ' '.join(x))

    X_train_tfidf = tfidf_vectorizer.fit_transform(df_train['Desc']).toarray()
    X_test_tfidf = tfidf_vectorizer.transform(df_test['Desc']).toarray()
    # X_val_tfidf = tfidf_vectorizer.transform(df_dev['Desc']).toarray()

    y_train = df_train['label'].values
    y_test = df_test['label'].values
    # y_val = df_dev['label'].values

    # return X_train_tfidf, X_test_tfidf, X_val_tfidf, y_train, y_test, y_val
    return X_train_tfidf, X_test_tfidf, y_train, y_test

def vectorize_Kfold(df):
    tfidf_vectorizer = TfidfVectorizer()
    
    df['Desc'] = df['Desc'].apply(lambda x: ' '.join(x))
    X = tfidf_vectorizer.fit_transform(df['Desc']).toarray()
    y = df['label'].values
    
    with open('src/models/vectorizer/tfidf_vectorizer.pkl', 'wb') as file:
        pickle.dump(tfidf_vectorizer, file)

    return X, y, tfidf_vectorizer

def initialise_model(hidden_units, num_classes, vocab_size, dropout, activation, lr, l2_reg):
    OPTIMIZER = tf.keras.optimizers.Adam(lr)
    # embedding_dim = 200
    # max_length = 2898

    # input_layer = Input(shape=(max_length,))
    # embedding_layer = Embedding(input_dim=vocab_size, output_dim=embedding_dim)(input_layer)
    # conv_layers = []
    # kernel_sizes = [2, 3, 4]
    # for kernel_size in kernel_sizes:
    #     conv_layer = Conv1D(filters=100, kernel_size=kernel_size, activation='relu')(embedding_layer)
    #     conv_layers.append(GlobalMaxPooling1D()(conv_layer))
    # concat_layer = Concatenate()(conv_layers)
    # dense_layer = Dense(10, activation='relu')(concat_layer)
    # output_layer = Dense(num_classes, activation='softmax')(dense_layer)
    # model = Model(inputs=input_layer, outputs=output_layer)

    model = tf.keras.Sequential([
        # tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=200),
        # tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(hidden_units, return_sequences=True)),
        # tf.keras.layers.Conv1D(filters=16, kernel_size=2, activation=activation),
        # tf.keras.layers.GlobalMaxPooling1D(),
        # tf.keras.layers.Dense(hidden_units, activation=activation),
        # tf.keras.layers.Dropout(dropout),
        # tf.keras.layers.BatchNormalization(),
        # tf.keras.layers.Dense(hidden_units//2, activation=activation),
        
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

def train_Kfold(model, df, X, y, NUM_EPOCHS, BATCH_SIZE, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    hist_list = []
    models = []
    fold_var = 1
    # model = create_model(X)
    for fold, (train_idx, val_idx) in enumerate(skf.split(df.index, df['label'])):
        print(f"\nTraining on Fold {fold_var}\n--------------------------------")
        # model = create_model(X)
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        
        hist = model.fit(
            X_train_fold, y_train_fold,
            batch_size=BATCH_SIZE,
            epochs=NUM_EPOCHS,
            validation_data=(X_val_fold, y_val_fold),
            callbacks=[early_stop]
        )
        
        hist_list.append(hist)
        models.append(model)
        output_dir = "src/models/KFold_models"
        weights_path = f'{output_dir}/model_fold_{fold_var}.weights.h5'
        model.save_weights(weights_path)
        fold_var += 1
    return hist_list, models

def create_model(X):
    return initialise_model(hidden_units=256,
                                num_classes=5,
                                vocab_size=X.shape[1],
                                dropout=.3,
                                activation='leaky_relu',
                                lr=1e-4,
                                l2_reg=1e-4)
