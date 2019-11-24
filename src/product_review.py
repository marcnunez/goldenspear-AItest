import string
import pandas as pd

from src import memory
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import numpy as np
import nltk
from src import models

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


def concat_tokens_from_rating(review_df):
    data = {k: [] for k in range(1,6)}
    dict_text = {
        1:[],2:[],3:[],4:[],5:[]
    }
    for row, col in review_df.iterrows():
        dict_text[col['RATING']] = dict_text[col['RATING']] + col['TEXT']
    return dict_text


def print_frequencies(dict_ratings):
    for i in range(1, 6):
        nlp = nltk.FreqDist(dict_ratings[i])
        title = "Rating: " + str(i)
        nlp.plot(5, title=title)


def main():
    in_path = '../dataset/ratings.csv'

    review_df = read_csv(in_path)
    print(np.unique(review_df['RATING'], return_counts=True))

    before_token = clean_text(review_df)
    review_df = tokenizer(before_token)
    review_df = remove_stop_words(review_df)
    print(review_df.head())

    dict_ratings = concat_tokens_from_rating(review_df)
    print_frequencies(dict_ratings)

    X = before_token['TEXT']
    y = before_token['RATING']

    models.log_regression(X, y)
    models.naive_bayes_classifier(X, y)
    models.sgd_classifier(X, y)


if __name__ == '__main__':
    main()