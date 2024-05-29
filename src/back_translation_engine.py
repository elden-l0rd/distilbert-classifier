import pandas as pd
from modules.preprocessing import basic_processing
from modules.translate import back_translation

from modules import preprocessing as pre
from modules import extract_keywords as ek
from modules import visualise as vis
import models.model as model

# Section to create new translations
# target = 'af' # Afrikaans
target = 'mt' # Maltese
# target = 'bn' # Bengali
# target = 'kn' # Kannada
'''
PATH = 'data/external/mitre-classified.xlsx'
df = pd.read_excel(PATH)

# Basic cleaning of data
df = basic_processing(df)
df['NameDesc_original'] = df['NameDesc']
df['NameDesc'] = df['NameDesc'].apply(lambda x: back_translation(text=x, target=target).text)
df.to_excel(f'data/results/translated_{target}.xlsx', index=False)
'''

# Train model
PATH = f'data/results/translated_{target}.xlsx'
df = pd.read_excel(PATH)

# train test dev split
df_train, df_test, df_dev = pre.split_data(df, train_set_size=0.3, test_set_size=0.7, dev=True)

col_toDrop = ['Ref', 'Name', 'Desc', 'Confidentiality', 'Integrity', 'Availability', 'Ease Of Exploitation', 'References', 'Unnamed: 0']
df_train = df_train.reset_index(drop=True).drop(columns=col_toDrop)
df_test = df_test.reset_index(drop=True).drop(columns=col_toDrop)
df_dev = df_dev.reset_index(drop=True).drop(columns=col_toDrop)
print("Data split:")
print(f"df_train:\n{df_train['STRIDE'].value_counts()}\n")
print(f"df_dev:\n{df_dev['STRIDE'].value_counts()}\n")
print(f"df_test:\n{df_test['STRIDE'].value_counts()}")
print("=========================================\n")

# trivially extract keywords
df_train = pre.text_preprocessing(df_train)
df_test = pre.text_preprocessing(df_test)
df_dev = pre.text_preprocessing(df_dev)
print("Trivial text preprocessing:")
print(f"df_train:\n{df_train.head(1)}\n")

# obtain better set of keywords
df_train = ek.better_keywords(df_train)
vis.plot_wc(df_train)

X_train_tfidf, X_test_tfidf, X_val_tfidf, y_train, y_test, y_val = model.vectorize(df_train, df_test, df_dev)

NUM_EPOCHS = 100
BATCH_SIZE = 16
CLASSES = [0,1,2,3,4,5]
DROPOUT = .23

model8 = model.initialise_model(hidden_units=128,
                               num_classes=6,
                               vocab_size=X_train_tfidf.shape[1],
                               dropout=DROPOUT,
                               activation='leaky_relu',
                               lr=1e-3,
                               l2_reg=1e-4)

hist6, model8 = model.train_loop(model=model8,
                                 X_train_tfidf=X_train_tfidf,
                                 y_train=y_train,
                                 X_val_tfidf=X_val_tfidf,
                                 y_val=y_val,
                                 NUM_EPOCHS=NUM_EPOCHS,
                                 BATCH_SIZE=BATCH_SIZE)

vis.plot_data(hist=hist6,
               model=model8,
               X_val_padded=X_test_tfidf,
               y_val=y_test,
               classes=CLASSES)