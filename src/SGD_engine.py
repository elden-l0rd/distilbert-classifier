'''
Implements Stochastic Gradient Descent (SGD) Classifier for the CAPEC dataset.

PART 1: Train a SGD classifier model
PART 2: Load the model and perform prediction on the specified dataset

Results are at the end of the code.
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from modules import preprocessing as pre
# from modules import extract_keywords as ek
from modules import visualise as vis
import hyperparm as hpt
from numpy import random
import nltk
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer, TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report, make_scorer, f1_score
import matplotlib.pyplot as plt
nltk.download('stopwords')
from nltk.corpus import stopwords
import re
import joblib


'''
Each category is represented by a number:
S   T   I   D   E
|   |   |   |   |
0   1   2   3   4
'''

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

if __name__ == '__main__':
    '''
    PART 1: Train a SGD classifier model
    '''
    # PATH = 'data/external/raw/raw_capec_data_BERT.xlsx'
    # df = pd.read_excel(PATH)
    # label_changes = {3: 2, 4: 3, 5: 4}
    # df['label'] = df['label'].replace(label_changes)
    # df['Desc'] = df['Desc'].apply(clean_text)
    # print(df['label'].value_counts().sort_index())
    
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
    #                                   tol=None)),])
    
    # sgd.fit(X_train, y_train)
    # y_pred = sgd.predict(X_test)
    
    # print(classification_report(y_test, y_pred,target_names=['0','1','2','3','4']))
    
    # conf_matrix = confusion_matrix(y_test, y_pred, labels=[0, 1, 2, 3, 4])
    # print("Confusion Matrix:")
    # print(conf_matrix)
    # f1 = f1_score(y_test, y_pred, average='weighted')
    # print("F1 Score:", f1)
    
    # joblib.dump(sgd, 'src/models/SGD/sgd_model.pkl')  # save the model
    
    '''
    PART 2: Load the model and perform prediction on the specified dataset
    '''
    df = pd.read_excel('data/external/dataset.xlsx')
    df['Desc'] = df['Desc'].apply(clean_text)
    sgd = joblib.load('src/models/SGD/sgd_model.pkl')

    predicted_labels = sgd.predict(df['Desc'])
    df['predicted_label'] = predicted_labels

    results = {}
    for label in df['label'].unique():
        correct = ((df['label'] == label) & (df['label'] == df['predicted_label'])).sum()
        total = (df['label'] == label).sum()
        accuracy = correct / total if total != 0 else 0
        results[label] = (correct, total, accuracy)
    print("Classification results:")
    for label, (correct, total, acc) in results.items():
        print(f"Label {label}: {correct}/{total} correctly classified. Accuracy: {acc:.2f}")


    
    ################################################
    '''
    This section uses cross validation to find the best hyperparameters for the SGD classifier.
    '''
    # from sklearn.model_selection import GridSearchCV
    # param_grid = {
    #     'clf__alpha': (1e-6, 5e-5, 3e-5, 1e-5, 5e-4, 3e-5, 1e-4, 5e-3, 3e-3, 1e-3, 1e-2),
    #     'clf__max_iter': (10, 20, 50, 80),
    #     'clf__loss': ['hinge', 'log_loss', 'squared_error', 'huber',
    #                     'epsilon_insensitive', 'squared_epsilon_insensitive',
    #                     'squared_hinge', 'modified_huber', 'perceptron'],
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


'''
Results from running SGDClassifier:

label
0     78
1    145
2    123
3     40
4    159
Name: count, dtype: int64

              precision    recall  f1-score   support

           0       0.80      0.67      0.73        18
           1       0.75      0.88      0.81        24
           2       0.88      0.88      0.88        24
           3       0.78      0.70      0.74        10
           4       0.79      0.79      0.79        33

    accuracy                           0.80       109
   macro avg       0.80      0.78      0.79       109
weighted avg       0.80      0.80      0.80       109

Confusion Matrix:
[[12  1  2  1  2]
 [ 0 21  0  1  2]
 [ 0  0 21  0  3]
 [ 2  1  0  7  0]
 [ 1  5  1  0 26]]
F1 Score: 0.7967334452124408
'''
