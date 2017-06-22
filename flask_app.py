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
    return render_template("index.html")

@app.route('/slides')
def slides():
    return render_template("slides.html")

@app.route('/input')
def input():
    return render_template("input.html")

@app.route('/inputSearch')
def inputSearch():
    return render_template("inputSearch.html")

def sentiment(sentence):
    vs = analyzer.polarity_scores(sentence)
    return vs

@app.route('/output')
def get_user_data():
    # this is what the user entered
    screen_name = request.args.get('screen_name')
    try:
        # get last 100 user tweets
        twts = api.user_timeline(screen_name = screen_name, count = 100, include_rts = True, tweet_mode = "extended")



 # this is for taking only the tweets with NO mentions
        tweetsText = ''
        tweets = []
        for tweet in twts:
            # pull date off before getting _json
            created_at = tweet.created_at.strftime('%m/%d/%Y')
            tweet = tweet._json #First convert tweets from string to json object.

            # collect my mentions & process
            entities = tweet['entities']
            mentions = entities['user_mentions']
            mentions = re.sub(r'[^\x00-\x7f]',r'', str(mentions))

            #only keep if empty
            if mentions == "[]":
                text = tweet['full_text']
                tweets.append([created_at, text])
                tweetsText += ' ' + text + ' '

        vectorizer, forest = joblib.load(basepath + '/trolltrackrapp/model/forest_model.pkl')
        text = vectorizer.transform([tweetsText])

        predicted = forest.predict(text)
        probs = forest.predict_proba(text)

        if predicted[0]==1:
            the_result = "{x} IS at risk for engaging in trolling. Trolling probability = {y}".format(x=screen_name,y=round(probs[0][1],2))
        else:
            the_result = "{x} is NOT at risk for engaging in trolling. Trolling probability = {y}".format(x=screen_name,y=round(probs[0][1],2))


    except tweepy.TweepError as e:
        the_result = "Sorry, not a valid user name."

    the_result = the_result
    recent_tweets = tweets
    return render_template("output.html", the_result = the_result , recent_tweets = recent_tweets)





