from sklearn.base import TransformerMixin, BaseEstimator
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np
import pandas as pd



'''Function to process stopwords'''
def stop_words(text, language='english'):
    stop_words = set(stopwords.words(language))
    word_tokens = word_tokenize(text)
    text = [w for w in word_tokens if w not in stop_words]
    return ' '.join(text)



'''Function Transform for text manipulation

Remove HTML tags
Remove mail
Remove extra whitespaces
Convert accented characters to ASCII characters
Expand contractions
Remove special characters
Convert number words to numeric form
Remove numbers

'''
class TextPreprocessing(TransformerMixin, BaseEstimator):
    def __init__(self,
                 drop_digit=False,
                 drop_email=False,
                 drop_stop_words=False,
                 stop_words_language='english',
                 to_lemmatize=False
                 ):
        self.drop_digit = drop_digit
        self.drop_email = drop_email
        self.drop_stop_words = drop_stop_words
        self.stop_words_language = stop_words_language
        self.to_lemmatize = to_lemmatize


    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):

        X = X.str.replace('[^\w\s\d]', '', regex=True).str.lower()

        if self.drop_email:
            X = X.apply(lambda x: ' '.join([
                text.replace('^[a-z0-9]+[\._]?[a-z0-9]+[@]\w+[.]\w{2,3}$', '',regex=True)
                for text in word_tokenize(x)
            ]))


        if self.drop_digit:
            X = X.str.replace('\d', '', regex=True)

        if self.drop_stop_words:
            X = X.apply(lambda x: stop_words(x, self.stop_words_language))

        if self.to_lemmatize:
            X = X.apply(lambda x: ' '.join([
                WordNetLemmatizer().lemmatize(word)
                for word in word_tokenize(x)
            ]))

        return X
