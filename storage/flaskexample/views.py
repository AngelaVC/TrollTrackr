from .a_Model import ModelIt
from flask import render_template
from flask import request
from flaskexample import app
import pandas as pd
from datetime import date
from sklearn.externals import joblib

from nltk.sentiment.vader import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

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


@app.route('/')
@app.route('/index')
def index():
    return render_template("index.html",
       title = 'Home'
       )

@app.route('/input')
def input():
    return render_template("input.html")

@app.route('/output')



def get_user_data():     
    #keys = ['description','favourites_count', 'followers_count', 'listed_count',
    #   'friends_count', 'statuses_count','created_at']
    
    # this is what the user entered
    screen_name = request.args.get('screen_name')
    
    try:
        # get last 50 user tweets
        twts = api.user_timeline(screen_name = screen_name, count = 100, include_rts = True, tweet_mode = "extended")  
        
        # text of each tweet
        tweets = [[tweet.full_text] for tweet in twts]
        tweetsText = ''
        for tweet in tweets:
            tweetsText += ' ' + tweet[0]
        
        #d = user._json  # just pull out the json bit
        ## put the data in smaller dictionary 
        #dd = {key: d[key] for key in keys }
        # pull out the description and run the sentiment function
        #s = analyzer.polarity_scores(dd['description'])
        #dd['compound'] = s['compound']
        # turn them into timestamps
        #dd['created_at'] = pd.to_datetime(dd['created_at'])
        #calc = date.today() - dd['created_at'].date()
        #dd['tweets_per_day'] = dd['statuses_count']/calc.days
        #model = joblib.load('.pkl')
        #X = [1, dd['favourites_count'], dd['followers_count'], dd['listed_count'],dd['tweets_per_day'], dd['friends_count'], dd['statuses_count'], dd['compound']]
        
        #with open('forest_model.pkl', 'rb') as fin:
        #    vectorizer, forest = pickle.load(fin)
        #    xx = pickle.load(fin)
        
        vectorizer, forest = joblib.load('forest_model.pkl')
            
        text = vectorizer.transform([tweetsText])

        
        predicted = forest.predict(text)
        probs = forest.predict_proba(text)
        
        if predicted[0]==1: 
            the_result = "{x} IS at risk for engaging in trolling. Trolling probability = {y}".format(x=screen_name,y=probs[0][1])
        else:
            the_result = "{x} is NOT at risk for engaging in trolling. Trolling probability = {y}".format(x=screen_name,y=probs[0][1])
            
    except tweepy.TweepError as e:
        the_result = "Sorry, not a valid user name."
    
    the_result = the_result
    return render_template("output.html", the_result = the_result,recentTweets = tweets)
