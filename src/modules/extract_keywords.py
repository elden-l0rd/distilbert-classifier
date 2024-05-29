from modules import preprocessing as pre
from .predefined_keywords import *
import gensim.downloader
from tqdm import tqdm

MIN_COSINE_VALUE = 0.28

def load_word2vec_model():
    with tqdm(total=1, desc="Loading Word2Vec Model", unit="model") as pbar:
        w2v_model = gensim.downloader.load('word2vec-google-news-300')
        pbar.update(1)
    return w2v_model
w2v = load_word2vec_model()

def merge_lists(l1, min_cosine_value, w2v, ref_list):
    final_list = []
    
    for word in l1:
        if word in w2v:
            similarities = []
            for ref_word in ref_list:
                if ref_word in w2v:
                    sim = w2v.similarity(word, ref_word)
                    similarities.append(sim)
            if any(sim >= min_cosine_value for sim in similarities):
                final_list.append(word)
    if 'use' in final_list: final_list.remove('use')
    elif 'also' in final_list: final_list.remove('also')
    return final_list

def better_keywords(df):
    for i in range(len(df)):
        if df.iloc[i]['label'] == 0:
            updated_list = merge_lists(df.loc[i, 'Desc'], MIN_COSINE_VALUE, w2v, S_final)
        elif df.iloc[i]['label'] == 1:
            updated_list = merge_lists(df.loc[i, 'Desc'], MIN_COSINE_VALUE, w2v, E_final)
        elif df.iloc[i]['label'] == 2:
            updated_list = merge_lists(df.loc[i, 'Desc'], MIN_COSINE_VALUE, w2v, D_final)
        elif df.iloc[i]['label'] == 3:
            updated_list = merge_lists(df.loc[i, 'Desc'], MIN_COSINE_VALUE, w2v, I_final)
        elif df.iloc[i]['label'] == 4:
            updated_list = merge_lists(df.loc[i, 'Desc'], MIN_COSINE_VALUE, w2v, R_final)
        else:
            updated_list = merge_lists(df.loc[i, 'Desc'], MIN_COSINE_VALUE, w2v, T_final)
        
        df.at[i, 'Desc'] = updated_list
    return df
