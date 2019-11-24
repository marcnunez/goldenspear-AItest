import string
import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src import memory
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import numpy as np


def read_csv(path: str)->pd.DataFrame:
    df = pd.read_csv(path)
    return df


@memory.cache()
def clean_text(review_df: pd.DataFrame)->pd.DataFrame:
    review_df['TEXT'] = review_df['TEXT'].apply(lambda x: clean_punctuation(x.lower()))
    review_df['TEXT'] = review_df['TEXT'].apply(lambda x: clean_digits(x))
    return review_df


# Clean digits
def clean_digits(review_text: str)->str:
    review_text = "".join([i for i in review_text if not i.isdigit()])
    return review_text


# Clean punctuation and symbols
def clean_punctuation(review_text: str)->str:
    review_text = "".join([i for i in review_text if i not in string.punctuation])
    return review_text


@memory.cache()
def tokenizer(review_df: pd.DataFrame)->pd.DataFrame:
    tokenizer = RegexpTokenizer(r'\w+')
    review_df['TEXT'] = review_df['TEXT'].apply(lambda x: tokenizer.tokenize(x))
    return review_df


def remove_stop_words_sentece(review_text: str)->str:
    words = [w for w in review_text if w not in stopwords.words('english')]
    return words


@memory.cache()
def remove_stop_words(review_df: pd.DataFrame)->pd.DataFrame:
    review_df['TEXT'] = review_df['TEXT'].apply(lambda x: remove_stop_words_sentece(x))
    return review_df


def one_hot_encoder(review_df: pd.DataFrame):
    values = np.array(review_df['TEXT'])

def get_top_n_words(corpus, n=None):
    vec = CountVectorizer().fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return words_freq[:n]


def main():
    in_path = '../dataset/ratings.csv'

    review_df = read_csv(in_path)
    print(np.unique(review_df['RATING'], return_counts=True))

    review_df = clean_text(review_df)
    review_df = tokenizer(review_df)
    review_df = remove_stop_words(review_df)
    print(review_df.head())



    X = review_df['TEXT']
    y = review_df['RATING']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    sgd = Pipeline([('vect', CountVectorizer()),
                    ('tfidf', TfidfTransformer()),
                    ('clf',
                     SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42)),
                    ])
    sgd.fit(X_train, y_train)


    y_pred = sgd.predict(X_test)

    print('accuracy %s' %accuracy_score(y_pred, y_test))
    print(classification_report(y_test, y_pred))


if __name__ == '__main__':
    main()