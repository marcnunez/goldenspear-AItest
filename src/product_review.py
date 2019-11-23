import string
import pandas as pd
from src import memory


def read_csv(path: str)->pd.DataFrame:
    df = pd.read_csv(path)
    return df


@memory.cache()
def clean_text(review_df: pd.DataFrame)->pd.DataFrame:
    review_df['TEXT'] = review_df['TEXT'].apply(lambda  x: clean_punctuation(review_df['TEXT']))
    #review_df['TEXT'] = review_df['TEXT'].apply(lambda  x: clean_digits(review_df['TEXT']))
    return review_df


# Clean digits
def clean_digits(review_text: str)->str:
    review_text = "".join([i for i in review_text if not i.isdigit()])
    review_text = "".join([i for i in review_text if i not in string.punctuation])

    return review_text


# Clean punctuation and symbols
def clean_punctuation(review_text: str)->str:
    review_text = "".join([i for i in review_text if i not in string.punctuation])
    return review_text



def main():
    in_path = '../dataset/ratings.csv'

    review_df = read_csv(in_path)
    print(review_df.head())
    review_df = clean_text(review_df.head())
    print(review_df.head())

if __name__ == '__main__':
    main()