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

analyzer = SentimentIntensityAnalyzer()


# create the application object
app = Flask(__name__)
basepath = os.path.abspath(".")

app.config["DEBUG"] = True

# -------------------
# tweepy basic setup
import tweepy
from tweepy import OAuthHandler

consumer_key = 'qvcxO1WLtia8hdYVerM7NQb6o'
consumer_secret = 'FrH8NhxPyPCRtY2MqCmTtabIBmsVc9lNVM4NVSthSALiXJ5IaA'
access_token = '18085200-kCr3a2D0DM5vhD9Q2AFxKmkLVSoyKnI4fcsNcqvax'
access_secret = 'edgXlBlOwFG6JOjotTw18H5at8yOH1XwsZ5umMTGIpFyA'

auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)

api = tweepy.API(auth)

# -------------------


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


def sentiment(sentence):
    vs = analyzer.polarity_scores(sentence)
    return vs

@app.route('/output')
def get_user_data():
    # this is what the user entered
    screen_name = request.args.get('screen_name')
    try:
        # get last 200 user tweets
        twts = api.user_timeline(screen_name = screen_name, count = 200, include_rts = True, tweet_mode = "extended")



        # this is for taking all tweets, with and without mentions
        tweetsMentions = []
        tweetsNM = []
        for tweet in twts:
            # pull date off before getting _json
            created_at = tweet.created_at.strftime('%m/%d/%Y')

            tweet = tweet._json #First convert tweets from string to json object.

            # collect my mentions & process
            entities = tweet['entities']
            mentions = entities['user_mentions']
            mentions = re.sub(r'[^\x00-\x7f]',r'', str(mentions))

            text = tweet['full_text']
            # move to different places if empty or not
            if mentions != "[]":
                tweetsMentions.append([created_at, text])
            else:
                tweetsNM.append([created_at,text])

            NMtweetsText = ''
            for tweet in tweetsNM:
                NMtweetsText += ' ' + tweet[1] + ' '

        vectorizer, forest = joblib.load(basepath + '/trolltrackrapp/model/LR_model.pkl')
        text = vectorizer.transform([NMtweetsText])

        predicted = forest.predict(text)

        # look for you forms
        youForms = ['you','your','yours','yourself','yourselves']
        mentionTweets = pd.DataFrame(tweetsMentions,columns=['date','tweet'])
        mentionTweets['contains_you']=mentionTweets['tweet'].str.contains('|'.join(youForms))

        # with negative sentiment
        mentionTweets['sentiment']=mentionTweets['tweet'].apply(sentiment)
        mentionTweets = pd.concat([mentionTweets.drop(['sentiment'], axis=1), mentionTweets['sentiment'].apply(pd.Series)], axis=1)

        # together they give possible trolling
        possibleTrolling = mentionTweets[(mentionTweets['contains_you']==1)&(mentionTweets['compound']<0)]
        tweetsTrolling = possibleTrolling[['date','tweet']].values.tolist()
        if predicted[0]==1:
            if len(tweetsTrolling)==0:
                the_result = "{x} IS at risk for engaging in trolling, but has not trolled recently. Risk assessment based on tweets below. <a href=\"http://twitter.com/{x}\">Click here</a> to view @{x} on twitter and from there you can block or mute @{x}".format(x=screen_name)
                recent_tweets = tweetsNM
            else:
                the_result = "{x} IS at risk for engaging in trolling. Tweets that may be trolling are listed below.  <a href=\"http://twitter.com/{x}\">Click here</a> to view @{x} on twitter and block, mute, or report @{x}".format(x=screen_name)
                recent_tweets = tweetsTrolling

        else:
            the_result = "{x} is NOT at risk for engaging in trolling. Result based on recent tweets below.".format(x=screen_name)
            recent_tweets = tweetsNM

    except tweepy.TweepError as e:
        return render_template("output.html", the_result = "Sorry, not a valid user or user has no tweets.", recent_tweets = [['','']])

    the_result = the_result
    recent_tweets = recent_tweets
    return render_template("output.html", the_result = the_result , recent_tweets = recent_tweets)





