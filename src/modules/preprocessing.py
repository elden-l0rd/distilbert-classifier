import re
import random
import nltk
from nltk import SnowballStemmer
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
import neattext.functions as nfx
from sklearn.model_selection import train_test_split

def basic_processing(df):
    words_to_remove = ["e.g.", "code", "may", "attack", "system", "adversary", "Adversaries"]
    for word in words_to_remove:
        df['Desc'] = df['Desc'].apply(lambda x: str(x).replace(word, ''))
    for word in words_to_remove:
        df['Desc'] = df['Desc'].apply(lambda x: re.sub(r'\b' + re.escape(word) + r'\b', '', x))

    # df['Desc'] = df['Desc'].str.replace(r"\b(" + "|".join(words_to_remove) + r")\b", "", regex=True)
    df['Desc'] = df['Desc'].str.replace("<br><br>", "", regex=True)
    df['Desc'] = df['Desc'].str.replace("\(Citation:.*?\)", "", regex=True)
    df['Desc'] = df['Desc'].str.replace("http\S+", "", regex=True)
    df['Desc'] = df['Desc'].str.replace("  +", " ", regex=True)
    df['Desc'] = df['Desc'].str.replace("[^A-Za-z]", " ", regex=True)
    return df

def rm_stopwords(df):
    stop_words = set(stopwords.words('english'))
    df['Desc'] = df['Desc'].apply(lambda x: [word for word in x if word not in stop_words])
    # print(f"Removed stopwords:\n {df.head(3).Desc}\n")
    return df

def lemmatize(df):
    lemmatizer = WordNetLemmatizer()
    def lemmatize_tokens(tokens):
        def get_wordnet_pos(word):
            tag = nltk.pos_tag([word])[0][1][0].upper()
            tag_dict = {"J": wordnet.ADJ,
                        "N": wordnet.NOUN,
                        "V": wordnet.VERB,
                        "R": wordnet.ADV}
            return tag_dict.get(tag, wordnet.NOUN)
        lemmas = [lemmatizer.lemmatize(token, get_wordnet_pos(token)) for token in tokens]
        return lemmas
    df['Desc'] = df['Desc'].apply(lambda x: lemmatize_tokens(x))
    # print(f"Lemmatized words:\n {df.head(3).Desc}")
    return df

def text_preprocessing(df):
    basic_processing(df)
    df['Desc'] = df['Desc'].apply(lambda x: word_tokenize(x))
    rm_stopwords(df)
    lemmatize(df)

    k = random.randint(0, len(df)) # arbitary row to show that words have been removed
    print(f"Bef rm duplicates: {len(df.iloc[k]['Desc'])}")
    df['Desc'] = df['Desc'].apply(lambda x: list(set([word.lower() for word in x]))) # to remove duplicates
    print(f"Aft rm duplicates: {len(df.iloc[k]['Desc'])}")
    print(f"Removed duplicates:\n {df.head(3).Desc}")

    print("=========================================")
    return df

def split_data(df, train_set_size, test_set_size, dev, symmetric=False):
    '''
    train_set_size + test_set_size + dev_set_size = 1

    symmetric: Set True for ovr_engine, there needs to have equal number of samples for
               training and testing dataset
               default: False
    '''
    
    if not dev: # no dev set
        while True:
            df_train, df_test = train_test_split(df, test_size=1-train_set_size)
            if symmetric:
                min_samples = min(len(df_train), len(df_test))
                if len(df_train) > min_samples:
                    df_train = df_train.sample(n=min_samples)
                elif len(df_test) > min_samples:
                    df_test = df_test.sample(n=min_samples)

                c = set([0, 1, 2, 3, 4])
                if set(df_train['label'].unique()) != c or \
                    set(df_test['label'].unique()) != c:
                        continue
                else: break
        return df_train, df_test
    else:
        while True:
            df_train, temp = train_test_split(df, test_size=1-train_set_size)
            df_test, df_dev = train_test_split(temp, test_size=test_set_size)

            c = set([0, 1, 2, 3, 4])
            if set(df_train['label'].unique()) != c or \
                set(df_test['label'].unique()) != c or \
                set(df_dev['label'].unique()) != c:
                    continue
            else: break
        return df_train, df_test, df_dev

def stemming(sentence):
    stemmer = SnowballStemmer("english")
    stemSentence = ""
    for word in sentence.split():
        stemSentence += stemmer.stem(word)
        stemSentence += " "
    stemSentence = stemSentence.strip()
    return stemSentence

def prepare_sentence(sentence):
    s = str(sentence).lower()
    s = nfx.remove_stopwords(s)
    s = stemming(s)
    #s = lemmatize(s)
    return s
