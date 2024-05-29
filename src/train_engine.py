import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from modules import preprocessing as pre
# from modules import extract_keywords as ek
from modules import visualise as vis
import models.model as model
import hyperparm as hpt

import logging
from numpy import random
import gensim
import nltk
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, make_scorer, f1_score
import matplotlib.pyplot as plt
nltk.download('stopwords')
from nltk.corpus import stopwords
import re
import joblib


'''
S   T   R   I   D   E
|   |   |   |   |   |
0   1   2   3   4   5
S   T   I   D   E
|   |   |   |   |
0   1   2   3   4
'''

def train_test():
    df_train, df_test = pre.split_data(df, train_set_size=0.75, test_set_size=0.7, dev=False)

    col_toDrop = ['Ref', 'Name', 'Desc', 'Confidentiality', 'Integrity', 'Availability', 'Ease Of Exploitation', 'References', 'Unnamed: 0']
    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)
    print("Data split:")
    print(f"df_train:\n{df_train['label'].value_counts()}\n")
    print(f"df_test:\n{df_test['label'].value_counts()}")
    print("=========================================\n")

    # trivially extract keywords
    df_train = pre.text_preprocessing(df_train)
    df_test = pre.text_preprocessing(df_test)
    print("Trivial text preprocessing:")
    print(f"df_train:\n{df_train.head(1)}\n")
    
    X_train_tfidf, X_test_tfidf, y_train, y_test = model.vectorize(df_train, df_test)

    NUM_EPOCHS = 50
    BATCH_SIZE = 16
    CLASSES = [0,1,2,3,4]
    DROPOUT = .2

    model6 = model.initialise_model(hidden_units=256,
                                num_classes=5,
                                vocab_size=X_train_tfidf.shape[1],
                                dropout=DROPOUT,
                                activation='relu',
                                lr=1e-3,
                                l2_reg=1e-4)

    hist6, model6 = model.train_loop(model=model6,
                                     X_train_tfidf=X_train_tfidf,
                                     y_train=y_train,
                                    #  X_val_tfidf=X_val_tfidf,
                                    #  y_val=y_val,
                                     NUM_EPOCHS=NUM_EPOCHS,
                                     BATCH_SIZE=BATCH_SIZE)
    
    f1_score = vis.plot_data(hist=hist6,
                model=model6,
                X_val_padded=X_test_tfidf,
                y_val=y_test,
                classes=CLASSES)
    
    return f1_score

def Kfold():
    df_clean = pre.text_preprocessing(df)
    print("Trivial text preprocessing:")
    X = df['Desc']
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # obtain better set of keywords
    # df_clean = ek.better_keywords(df_clean)
    # vis.plot_wc(df_clean)

    X, y, tfidf_vectorizer = model.vectorize_Kfold(df_clean)
    
    NUM_EPOCHS = 50
    BATCH_SIZE = 16
    DROPOUT = .3

    model9 = model.initialise_model(hidden_units=256,
                                num_classes=5,
                                vocab_size=X.shape[1],
                                dropout=DROPOUT,
                                activation='leaky_relu',
                                lr=1e-3,
                                l2_reg=1e-4)

    hist9, model9 = model.train_Kfold(model=model9,
                            df=df,
                            X=X,
                            y=y,
                            NUM_EPOCHS=NUM_EPOCHS,
                            BATCH_SIZE=BATCH_SIZE,
                            n_splits=5)

    f1_scores, _ = vis.plot_kfold(X=X, y=y, models=model9, n_splits=5)
    
    y_pred_test = model9[4].predict(pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen=200))
    y_pred_test = np.argmax(y_pred_test, axis=1)

    f1_test = f1_score(y_test, y_pred_test, average='weighted')
    accuracy_test = accuracy_score(y_test, y_pred_test)
    print(f"Test Set Accuracy: {accuracy_test:.2f}")
    print(f"Test Set F1 Score: {f1_test:.2f}")
    
    return f1_test, accuracy_test

def clean_text(text):
    REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
    BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
    STOPWORDS = set(stopwords.words('english'))
    
    text = str(text)
    text = text.lower()
    text = REPLACE_BY_SPACE_RE.sub(' ', text)
    text = BAD_SYMBOLS_RE.sub('', text)
    text = ' '.join(word for word in text.split() if word not in STOPWORDS)
    return text

def predict_sentence(model, sentence):
    cleaned_sentence = clean_text(sentence)
    prediction = model.predict([cleaned_sentence])
    return prediction[0]

if __name__ == '__main__':
    PATH = 'data/external/raw/raw_capec_data_BERT.xlsx'
    df = pd.read_excel(PATH)
    label_changes = {3: 2, 4: 3, 5: 4}
    df['label'] = df['label'].replace(label_changes)
    df['Desc'] = df['Desc'].apply(clean_text)
    print(df['label'].value_counts().sort_index())
    
    # X = df['Desc']
    # y = df['label']
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # sgd = Pipeline([('vect', CountVectorizer()),
    #             ('tfidf', TfidfTransformer()),
    #             ('clf', SGDClassifier(loss='hinge',
    #                                   penalty='l2',
    #                                   alpha=1e-3,
    #                                   random_state=42,
    #                                   max_iter=50,
    #                                   tol=None)),
    #            ])
    # sgd.fit(X_train, y_train)
    # y_pred = sgd.predict(X_test)
    
    # print('accuracy %s' % accuracy_score(y_pred, y_test))
    # print(classification_report(y_test, y_pred,target_names=['0','1','2','3','4']))
    
    # conf_matrix = confusion_matrix(y_test, y_pred, labels=[0, 1, 2, 3, 4])
    # print("Confusion Matrix:")
    # print(conf_matrix)
    # f1 = f1_score(y_test, y_pred, average='weighted')
    # print("F1 Score:", f1)
    
    # joblib.dump(sgd, 'src/models/SGD/sgd_model.pkl')
    
    # sgd = joblib.load('src/models/SGD/sgd_model.pkl')
    
    # df = pd.read_csv('data/external/unseen.csv')
    # df['Description'] = df['Description'].apply(clean_text)
    # sgd = joblib.load('src/models/SGD/sgd_model.pkl')

    # predicted_labels = sgd.predict(df['Description'])
    # df['predicted_label'] = predicted_labels

    # results = {}
    # for label in df['label'].unique():
    #     correct = ((df['label'] == label) & (df['label'] == df['predicted_label'])).sum()
    #     total = (df['label'] == label).sum()
    #     accuracy = correct / total if total != 0 else 0
    #     results[label] = (correct, total, accuracy)
    # print("Classification results:")
    # for label, (correct, total, acc) in results.items():
    #     print(f"Label {label}: {correct}/{total} correctly classified. Accuracy: {acc:.2f}")

    
    ################################################
    # from sklearn.model_selection import GridSearchCV
    # param_grid = {
    #     'clf__alpha': (1e-6, 5e-5, 3e-5, 1e-5, 5e-4, 3e-5, 1e-4, 5e-3, 3e-3, 1e-3, 1e-2),
    #     'clf__max_iter': (10, 20, 50, 80),
    #     'clf__loss': ['hinge', 'log_loss', 'squared_error', 'huber',
    #                   'epsilon_insensitive', 'squared_epsilon_insensitive',
    #                   'squared_hinge', 'modified_huber', 'perceptron'],
    #     'clf__penalty': ['l2', 'l1', 'elasticnet'],
    #     'vect__ngram_range': [(1, 1), (1, 2), (1, 3)],
    #     'tfidf__use_idf': (True, False),
    # }
    # grid_search = GridSearchCV(sgd, param_grid, cv=5, scoring='f1_weighted', verbose=1)
    # grid_search.fit(X_train, y_train)
    # y_pred = grid_search.predict(X_test)
    # print("Best parameters:", grid_search.best_params_)
    # print("Best cross-validation F1 score: {:.2f}".format(grid_search.best_score_))
    # y_pred = grid_search.predict(X_test)
    # print('Accuracy %s' % accuracy_score(y_pred, y_test))
    # print(classification_report(y_test, y_pred, target_names=['0', '1', '2', '3', '4']))
    # conf_matrix = confusion_matrix(y_test, y_pred, labels=[0, 1, 2, 3, 4])
    # print("Confusion Matrix:")
    # print(conf_matrix)
    # f1 = f1_score(y_test, y_pred, average='weighted')
    # print("F1 Score:", f1)

    ################################################
    
    # train_test()
    Kfold()
    
    # f1_values = []
    # for i in range(10):
    #     f1_scores = Kfold()
    #     f1sk = np.mean(f1_scores)
    #     f1s = train_test()
    #     f1_values.append((f1sk, f1s))
    # print(f'f1values: {f1_values}')

'''
Best parameters: {'clf__alpha': 5e-05, 'clf__loss': 'modified_huber', 'clf__max_iter': 10, 'clf__penalty': 'elasticnet'}
Best cross-validation score: 0.73
Accuracy 0.7339449541284404
              precision    recall  f1-score   support

           0       0.92      0.61      0.73        18
           1       0.70      0.79      0.75        24
           2       0.78      0.75      0.77        24
           3       0.82      0.90      0.86        10
           4       0.64      0.70      0.67        33

    accuracy                           0.73       109
   macro avg       0.77      0.75      0.75       109
weighted avg       0.75      0.73      0.73       109

Confusion Matrix:
[[11  1  1  0  5]
 [ 0 19  1  0  4]
 [ 0  2 18  0  4]
 [ 0  0  1  9  0]
 [ 1  5  2  2 23]]
F1 Score: 0.7342822040000854

Best parameters: {'clf__alpha': 3e-05, 'clf__loss': 'hinge', 'clf__max_iter': 50, 'clf__penalty': 'l2', 'tfidf__use_idf': True, 'vect__ngram_range': (1, 3)}
Best cross-validation F1 score: 0.75
Accuracy 0.7614678899082569
              precision    recall  f1-score   support

           0       0.92      0.61      0.73        18
           1       0.70      0.88      0.78        24
           2       0.83      0.79      0.81        24
           3       0.70      0.70      0.70        10
           4       0.74      0.76      0.75        33

    accuracy                           0.76       109
   macro avg       0.78      0.75      0.75       109
weighted avg       0.77      0.76      0.76       109

Confusion Matrix:
[[11  1  2  1  3]
 [ 0 21  0  1  2]
 [ 0  1 19  0  4]
 [ 1  2  0  7  0]
 [ 0  5  2  1 25]]
F1 Score: 0.760530161995022
'''