'''
Trains the feedforward model in models/feedforward_model.py
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from modules import preprocessing as pre
# from modules import extract_keywords as ek
from modules import visualise as vis
import models.feedforward_model as model
import hyperparm as hpt
import nltk
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
nltk.download('stopwords')
from nltk.corpus import stopwords
import re


'''
Each category is represented by a number:
S   T   I   D   E
|   |   |   |   |
0   1   2   3   4
'''

def train_test():
    df_train, df_test = pre.split_data(df, train_set_size=0.75, test_set_size=0.7, dev=False)

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
                                     NUM_EPOCHS=NUM_EPOCHS,
                                     BATCH_SIZE=BATCH_SIZE)
    
    f1_score = vis.plot_data(hist=hist6,
                model=model6,
                X_val_padded=X_test_tfidf,
                y_val=y_test,
                classes=CLASSES)
    
    return f1_score

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
    
    X = df['Desc']
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    train_test()
