import pandas as pd
from modules import preprocessing as pre
from modules import extract_keywords as ek
from modules import visualise as vis
import models.ovr_model as ovr
import hyperparm as hpt

PATH = 'data/external/mitre-classified.xlsx'
df = pd.read_excel(PATH)

'''
change labels for simplicity
S  T  R  I  D  E  (mapping)
0  5  4  3  2  1
'''

# train test split
df_train, df_test = pre.split_data(df, train_set_size=0.7, test_set_size=0.7, dev=False, symmetric=True)

col_toDrop = ['Ref', 'Name', 'Desc', 'Confidentiality', 'Integrity', 'Availability', 'Ease Of Exploitation', 'References', 'Unnamed: 0']
df_train = df_train.reset_index(drop=True).drop(columns=col_toDrop)
df_test = df_test.reset_index(drop=True).drop(columns=col_toDrop)
print("Data split:")
print(f"df_train:\n{df_train['STRIDE'].value_counts()}\n")
print(f"df_test:\n{df_test['STRIDE'].value_counts()}")
print("=========================================\n")

# trivially extract keywords
df_train = pre.text_preprocessing(df_train)
df_test = pre.text_preprocessing(df_test)
print("Trivial text preprocessing:")
print(f"df_train:\n{df_train.head(1)}\n")

# obtain better set of keywords
df_train = ek.better_keywords(df_train)
vis.plot_wc(df_train)

X_train_tfidf, X_test_tfidf, y_train, y_test = ovr.vectorize(df_train, df_test)

y_pred, acc, report, cm = ovr.train_loop(X_train_tfidf, y_train, y_test)
print("\nAccuracy:", acc)
print("Classification Report:", report)
print("Confusion Matrix:", cm)
vis.plot_cm(cm)