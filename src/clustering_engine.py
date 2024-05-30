import pandas as pd
import joblib
from modules import preprocessing as pre
from modules import extract_keywords as ek
from modules import visualise as vis
import models.clustering_models as clustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

PATH = 'data/external/raw/raw_capec_data_BERT.xlsx'
df = pd.read_excel(PATH)
label_changes = {3: 2, 4: 3, 5: 4}
df['label'] = df['label'].replace(label_changes)
df['Desc'] = df['Name'] + ' ' + df['Desc']

corpus = df['Desc'].apply(pre.prepare_sentence)
tfidf = TfidfVectorizer()

X_train_raw, X_test_raw, y_train, y_test = train_test_split(df['Desc'], df['label'], test_size=0.1, random_state=64)

tfidf.fit(X_train_raw.apply(pre.prepare_sentence))

X_train_tfidf = tfidf.transform(X_train_raw.apply(pre.prepare_sentence)).toarray()
X_test_tfidf = tfidf.transform(X_test_raw.apply(pre.prepare_sentence)).toarray()
print("Number of features (dimensions):", X_train_tfidf.shape[1])

X = tfidf.fit(corpus)
X = tfidf.transform(corpus).toarray()
y = df['label'].values
X_train_tfidf, X_test_tfidf, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=64)

# 0: Naive Bayes, 1: Random Forest, 2: Linear SVC
y_pred, accuracy, precision, recall, f1, cm = clustering.classifier_selector(2,
                                                                        X_train_tfidf,
                                                                        X_test_tfidf,
                                                                        y_train, y_test)
joblib.dump(tfidf, 'src/models/SVM/tfidf_vectorizer.joblib') # save the vectorizer

print("\nAccuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
vis.plot_cm(cm)

'''
0: F1-score: 0.665
1: F1-score: 0.746
2: F1-score: 0.853
'''