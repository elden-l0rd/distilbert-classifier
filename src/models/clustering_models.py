'''
Functions for Naive Bayes, Random Forest, and Linear SVC methods.

Select the classifier in NB_Randomforest_engine.py. The logic is controlled by classifier_selector().
'''

import joblib
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def naive_bayes(X_train_tfidf, X_test_tfidf, y_train, y_test):
    clf = MultinomialNB().fit(X_train_tfidf, y_train)
    y_pred = clf.predict(X_test_tfidf)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    cm = confusion_matrix(y_test, y_pred)
    return y_pred, accuracy, precision, recall, f1, cm

def random_forest(X_train_tfidf, X_test_tfidf, y_train, y_test):
    clf = RandomForestClassifier().fit(X_train_tfidf, y_train)
    y_pred = clf.predict(X_test_tfidf)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    cm = confusion_matrix(y_test, y_pred)
    return y_pred, accuracy, precision, recall, f1, cm

def linear_svc(X_train_tfidf, X_test_tfidf, y_train, y_test):
    clf = LinearSVC().fit(X_train_tfidf, y_train)
    y_pred = clf.predict(X_test_tfidf)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    cm = confusion_matrix(y_test, y_pred)
    
    joblib.dump(clf, 'src/models/SVM/linear_svc_model.joblib')
    return y_pred, accuracy, precision, recall, f1, cm

def classifier_selector(classifier, X_train_tfidf, X_test_tfidf, y_train, y_test):
    if classifier==0: return naive_bayes(X_train_tfidf, X_test_tfidf, y_train, y_test)
    elif classifier==1: return random_forest(X_train_tfidf, X_test_tfidf, y_train, y_test)
    elif classifier==2: return linear_svc(X_train_tfidf, X_test_tfidf, y_train, y_test)
    else: return "Invalid classifier"
