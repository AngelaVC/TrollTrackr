'''
This app uses twitter api to retrieve most recent tweets, then
determines whether user is troll or not using pickled model.
Output is the prediction for troll or not, plus either examples 
of trolling or the tweets that went into the model if there are no 
trolling examples
'''

# import the Flask class from the flask module
from flask import Flask, render_template, request
import pandas as pd
from datetime import date
from datetime import datetime

from sklearn.externals import joblib
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import ast
import re
import os

import config   # holds twitter api key

analyzer = SentimentIntensityAnalyzer()

# create the application object
app = Flask(__name__)
basepath = os.path.abspath(".")

app.config["DEBUG"] = True

# -------------------
# tweepy basic setup
import tweepy
from tweepy import OAuthHandler
import config

auth = OAuthHandler(config.consumer_key, config.consumer_secret)
auth.set_access_token(config.access_token, config.access_secret)

api = tweepy.API(auth)

# -------------------
# setting up all the pages for the app

# use decorators to link the function to a url
@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("input.html")

@app.route('/slides')
def slides():
    return render_template("slides.html")

@app.route('/input')
def input():
    return render_template("input.html")

@app.route('/home')
def home():
    return render_template("input.html")



# ------------------
# sentiment is part of the determination if user is troll
def sentiment(sentence):
    vs = analyzer.polarity_scores(sentence)
    return vs


# --------------------
# this function gets the tweets
#     splits into mentions and no mentions
#     with mentions data, rules are used to decide troll or not
#     with no mentions data, model is used to determine if troll
@app.route('/output')
def get_user_data():
    # this is what the user entered
    screen_name = request.args.get('screen_name')
    try:
        # get last 100 user tweets
        twts = api.user_timeline(screen_name = screen_name, count = 100, include_rts = True, tweet_mode = "extended")

        # this is for taking all tweets, with and without mentions
        tweets_mentions = []
        tweets_no_ment = []
        for tweet in twts:
            # pull date off before getting _json
            created_at = tweet.created_at.strftime('%m/%d/%Y')
            
            #convert tweets from string to json object
            tweet = tweet._json 

            # collect my mentions & process from each tweet
            entities = tweet['entities']
            mentions = entities['user_mentions']
            mentions = re.sub(r'[^\x00-\x7f]',r'', str(mentions))

            # grab text of each tweet
            text = tweet['full_text']
            
            # move to mentions / no_ment depending on if there are mentions
            if mentions != "[]":
                tweets_mentions.append([created_at, text])
            else:
                tweets_no_ment.append([created_at,text])

            nm_tweets_text = ''
            for tweet in tweets_no_ment:
                nm_tweets_text += ' ' + tweet[1] + ' '

        vectorizer, LR = joblib.load(basepath + 'trolltrackrapp/model/LR_model.pkl')
        text = vectorizer.transform([nm_tweets_text])

        predicted = LR.predict(text)

        # look for you forms from tweets with mentions
        you_forms = ['you','your','yours','yourself','yourselves']
        mention_df = pd.DataFrame(tweets_mentions,columns=['date','tweet'])
        mention_df['contains_you']=mention_df['tweet'].str.contains('|'.join(you_forms))

        # with negative sentiment
        mention_df['sentiment']=mention_df['tweet'].apply(sentiment)
        mention_df = pd.concat([mention_df.drop(['sentiment'], axis=1), mention_df['sentiment'].apply(pd.Series)], axis=1)

        # together they give trolling
        trolling_df = mention_df[(mention_df['contains_you']==1)&(mention_df['compound']<0)]
        trolling_list = trolling_df[['date','tweet']].values.tolist()
        
        # if app predicts a troll, return trolling tweets if they exist 
        # if there are none, say they are a troll and give tweets used to make determination
        if predicted[0]==1:
            if len(trolling_list)==0:
                the_result = "{x} IS at risk for engaging in trolling, but has not trolled recently. Risk assessment based on tweets below. <a href=\"http://twitter.com/{x}\">Click here</a> to view @{x} on twitter and from there you can block or mute @{x}".format(x=screen_name)
                recent_tweets = tweets_no_ment
            else:
                the_result = "{x} IS at risk for engaging in trolling. Tweets that may be trolling are listed below.  <a href=\"http://twitter.com/{x}\">Click here</a> to view @{x} on twitter and block, mute, or report @{x}".format(x=screen_name)
                recent_tweets = trolling_list
        
        # if app does not predict a troll, return recent tweets
        else:
            the_result = "{x} is NOT at risk for engaging in trolling. Result based on recent tweets below. below.".format(x=screen_name)
            recent_tweets = tweets_no_ment

    # return error if they did not enter valid user name
    except tweepy.TweepError as e:
        return render_template("output.html", the_result = "Sorry, not a valid user or user has no tweets.", recent_tweets = [['','']])

    the_result = the_result
    recent_tweets = recent_tweets
    return render_template("output.html", the_result = the_result , recent_tweets = recent_tweets)





