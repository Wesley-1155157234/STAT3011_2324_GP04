#Import dependancies
import numpy as np
import pandas as pd
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import subjectivity
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.util import *
import matplotlib.pyplot as plt
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import unicodedata

#Load Dataset
# Reading the saved data pickle file
df_stocks = pd.read_pickle('D:/work/STAT3011Pro2/data/pickled_ten_year_filtered_lead_para.pkl')
print(df_stocks.head())

#Convert adj close price into integer format
df_stocks['prices'] = df_stocks['adj close'].apply(np.int64)
# selecting the prices and articles
df_stocks = df_stocks[['prices', 'articles']]
print(df_stocks.head())

#Remove letfmost dots from news article headlines
df_stocks['articles'] = df_stocks['articles'].map(lambda x: x.lstrip('.-'))
print(df_stocks.head())

#Sentiment analysis
df = df_stocks[['prices']].copy()
# Adding new columns to the data frame
df["compound"] = ''
df["neg"] = ''
df["neu"] = ''
df["pos"] = ''

# Create SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()
# Download VADER lexicon
nltk.download('vader_lexicon')

for date, row in df_stocks.iterrows():
    try:
        # Sentiment analysis on the article
        sentence = unicodedata.normalize('NFKD', row['articles']).encode('ascii','ignore').decode('utf-8')
        ss = sid.polarity_scores(sentence)
        # Add the analysis result to the DataFrame
        df.at[date, 'compound'] = ss['compound']
        df.at[date, 'neg'] = ss['neg']
        df.at[date, 'neu'] = ss['neu']
        df.at[date, 'pos'] = ss['pos']
    except TypeError:
        print(row['articles'])
        print(date)

print(df.head())

df.to_csv('D:/work/STAT3011Pro2/data/siddata.csv', index=False)
