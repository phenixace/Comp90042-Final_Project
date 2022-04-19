import os
import re
import json
import nltk
import string
import enchant
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer 
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import CountVectorizer

class CleanData:
    @staticmethod
    def clean_at(data):
        return re.sub('@[^\s]+', ' ', data)

    @staticmethod
    def clean_URLs(data):
        return re.sub('((www\.[^\s]+)|(https?://[^\s]+))',' ',data)

    @staticmethod
    def clean_numbers(data):
        return re.sub('[0-9]+', '', data)

    @staticmethod
    def clean_non_english_word(data):
        d = enchant.Dict('en_US') # english dictionary
        return ' '.join([word for word in data.split() if d.check(word)])

    @staticmethod
    def clean_stopwords(data):
        STOPWORDS = set(stopwords.words('english'))
        return ' '.join([word for word in data.split() if word not in STOPWORDS])
        
    @staticmethod
    def clean_punctuations(data):
        punctuations_list = string.punctuation
        translator = str.maketrans('', '', punctuations_list)
        return data.translate(translator)