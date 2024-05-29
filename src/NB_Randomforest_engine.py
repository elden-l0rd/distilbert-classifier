import pandas as pd
import joblib
from modules import preprocessing as pre
from modules import extract_keywords as ek
from modules import visualise as vis
import models.ovr_model as ovr
import models.NB_Randomforest_model as NB_RF
import hyperparm as hpt
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


# X = tfidf.fit(corpus)
# X = tfidf.transform(corpus).toarray()
# y = df['label'].values
# X_train_tfidf, X_test_tfidf, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=64)

# train test split
# df_train, df_test = pre.split_data(df, train_set_size=0.7, test_set_size=0.3, dev=False)
# df_train = df_train.reset_index(drop=True)
# df_test = df_test.reset_index(drop=True)
# print("Data split:")
# print(f"df_train:\n{df_train['label'].value_counts()}\n")
# print(f"df_test:\n{df_test['label'].value_counts()}")
# print("=========================================\n")

# trivially extract keywords
df['Desc'] = df['Desc'].apply(pre.prepare_sentence)
# df_train = pre.text_preprocessing(df_train)
# df_test = pre.text_preprocessing(df_test)
# print("Trivial text preprocessing:")
# print(f"df_train:\n{df_train.head(1)}\n")

# obtain better set of keywords
# df_train = ek.better_keywords(df_train)
# vis.plot_wc(df_train)

# X_train_tfidf, X_test_tfidf, y_train, y_test = ovr.vectorize(df_train, df_test)

# 0: Naive Bayes, 1: Random Forest, 2: Linear SVC
y_pred, accuracy, precision, recall, f1, cm = NB_RF.classifier_selector(2,
                                                                        X_train_tfidf,
                                                                        X_test_tfidf,
                                                                        y_train, y_test)
joblib.dump(tfidf, 'src/models/SVM/tfidf_vectorizer.joblib')

print("\nAccuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
# vis.plot_cm(cm)

'''
F1-score: 0.853
'''
