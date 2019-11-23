import string
import pandas as pd
from src import memory
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords


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


def main():
    in_path = '../dataset/ratings.csv'

    review_df = read_csv(in_path)

    print(review_df.head())
    review_df = clean_text(review_df.head(1000))
    print(review_df.head())
    review_df = tokenizer(review_df.head(1000))
    print(review_df.head())
    review_df = remove_stop_words(review_df.head(1000))
    print(review_df.head())

if __name__ == '__main__':
    main()