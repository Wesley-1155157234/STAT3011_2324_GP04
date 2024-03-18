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

#Save result to a new file
#df.to_csv('./data/siddata.csv', index=False)

#Split training and testing data
train_start_date = '2007-01-01'
train_end_date = '2014-12-31'
test_start_date = '2015-01-01'
test_end_date = '2016-12-31'
train = df.loc[train_start_date : train_end_date]
test = df.loc[test_start_date:test_end_date]

#Split prediction labels for training and testing dataset
y_train = pd.DataFrame(train['prices'])
y_test = pd.DataFrame(test['prices'])


#Convert sentiment analysis score into numpy array
sentiment_score_list = []
for date, row in train.iterrows():
    # Extract the sentiment scores for the 'neg' and 'pos' columns
    sentiment_score = np.asarray([row['neg'], row['pos']])
    sentiment_score_list.append(sentiment_score)
numpy_df_train = np.asarray(sentiment_score_list)

sentiment_score_list = []
for date, row in test.iterrows():
    # Extract the sentiment scores for the 'neg' and 'pos' columns
    sentiment_score = np.asarray([row['neg'], row['pos']])
    sentiment_score_list.append(sentiment_score)
numpy_df_test = np.asarray(sentiment_score_list)


from treeinterpreter import treeinterpreter as ti
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import classification_report,confusion_matrix

rf = RandomForestRegressor()
rf.fit(numpy_df_train, y_train)

print(rf.feature_importances_)

prediction, bias, contributions = ti.predict(rf, numpy_df_test)
#print prediction
#print contributions
rf.score(numpy_df_test,y_test)

idx = pd.date_range(test_start_date, test_end_date)
predictions_df = pd.DataFrame(data=prediction[0:], index = idx, columns=['prices'])
predictions_df.head()
#Plot
ax = predictions_df.rename(columns=
                           {"prices": "predicted_price"}).plot(title=
                           'Random Forest predicted prices 8-2 years')
ax.set_xlabel("Dates")
ax.set_ylabel("Adj Close Prices")
fig = y_test.rename(columns={"prices": "actual_price"}).plot(ax = ax).get_figure()
fig.savefig("./graphs/random forest without smoothing.png")
# colors = ['332288', '88CCEE', '44AA99', '117733', '999933', 'DDCC77', 'CC6677', '882255', 'AA4499']

#Alignment of the testing dataset price value and pridicted values
from datetime import datetime, timedelta
temp_date = test_start_date
average_last_5_days_test = 0
total_days = 10
for i in range(total_days):
    average_last_5_days_test += test.loc[temp_date, 'prices']
    # Converting string to date time
    temp_date = datetime.strptime(temp_date, "%Y-%m-%d").date()
    # Reducing one day from date time
    difference = temp_date + timedelta(days=1)
    # Converting again date time to string
    temp_date = difference.strftime('%Y-%m-%d')
    #print temp_date
average_last_5_days_test = average_last_5_days_test / total_days
print(average_last_5_days_test)

temp_date = test_start_date
average_upcoming_5_days_predicted = 0
for i in range(total_days):
    average_upcoming_5_days_predicted += predictions_df.loc[temp_date, 'prices']
    # Converting string to date time
    temp_date = datetime.strptime(temp_date, "%Y-%m-%d").date()
    # Adding one day from date time
    difference = temp_date + timedelta(days=1)
    # Converting again date time to string
    temp_date = difference.strftime('%Y-%m-%d')
    #print temp_date
average_upcoming_5_days_predicted = average_upcoming_5_days_predicted / total_days
print(average_upcoming_5_days_predicted)
#average train.loc['2013-12-31', 'prices'] - advpredictions_df.loc['2014-01-01', 'prices']
difference_test_predicted_prices = average_last_5_days_test - average_upcoming_5_days_predicted
print(difference_test_predicted_prices)

predictions_df['prices'] = predictions_df['prices'] + difference_test_predicted_prices
predictions_df.head()

ax = predictions_df.rename(columns={"prices": "predicted_price"}).plot(title='Random Forest predicted prices 8-2 years after aligning')
ax.set_xlabel("Dates")
ax.set_ylabel("Stock Prices")
fig = y_test.rename(columns={"prices": "actual_price"}).plot(ax = ax).get_figure()
fig.savefig("./graphs/random forest with aligning.png")


#Smoothing based approach
#Applying EWMA pandas to smooth the stock prices
predictions_df['ewma'] = predictions_df["prices"].ewm(span=60).mean()
predictions_df.head()


predictions_df['actual_value'] = test['prices']
predictions_df['actual_value_ewma'] = predictions_df["actual_value"].ewm(span=60).mean()
predictions_df.head()

# Changing column names
predictions_df.columns = ['predicted_price', 'average_predicted_price', 'actual_price', 'average_actual_price']
# Now plotting test predictions after smoothing
predictions_plot = predictions_df.plot(title='Random Forest predicted prices 8-2 years after aligning & smoothing')
predictions_plot.set_xlabel("Dates")
predictions_plot.set_ylabel("Stock Prices")
fig = predictions_plot.get_figure()
fig.savefig("./graphs/random forest after smoothing.png")

# Plotting just predict and actual average curves
predictions_df_average = predictions_df[['average_predicted_price', 'actual_price']]
predictions_plot = predictions_df_average.plot(title='Random Forest 8-2 years after aligning & smoothing')
predictions_plot.set_xlabel("Dates")
predictions_plot.set_ylabel("Stock Prices")
fig = predictions_plot.get_figure()
fig.savefig("./graphs/random forest after smoothing 2.png")
