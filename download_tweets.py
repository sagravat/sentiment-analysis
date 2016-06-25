#!/usr/bin/env python
# encoding: utf-8

"""
    Sanjay Agravat

    Download Twitter data using a the Tweepy API to search for tweets by hashtag

        * modified from https://gist.github.com/yanofsky/5436496
        * uses a config file for the twitter API keys
        * processes and cleans tweets including removing AT_USER and retweets
        * saves the tweets to file
"""

import tweepy
import csv
import argparse
import json
import os
import re
import optparse
import sys


def parse_config():
    config = {}
    # from file args
    if os.path.exists('config.json'):
        with open('config.json') as f:
            config.update(json.load(f))
    else:
        # may be from command line
        parser = argparse.ArgumentParser()

        parser.add_argument('-ck', '--consumer_key', default=None, help='Your developper `Consumer Key`')
        parser.add_argument('-cs', '--consumer_secret', default=None, help='Your developper `Consumer Secret`')
        parser.add_argument('-at', '--access_token', default=None, help='A client `Access Token`')
        parser.add_argument('-ats', '--access_token_secret', default=None, help='A client `Access Token Secret`')

        args_ = parser.parse_args()

        def val(key):
            return config.get(key) \
                   or getattr(args_, key) \
                   or raw_input('Your developper `%s`: ' % key)

        config.update({
            'consumer_key': val('consumer_key'),
            'consumer_secret': val('consumer_secret'),
            'access_token': val('access_token'),
            'access_token_secret': val('access_token_secret'),
        })
    # should have something now
    return config

def process_tweet(tweet):

    # Convert to lower case
    tweet = tweet.lower()

    # Convert https?://* to URL
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'URL', tweet)

    # Convert @username to AT_USER
    tweet = re.sub('@[^\s]+', 'AT_USER', tweet)

    # Remove additional white spaces
    tweet = re.sub('[\s]+', ' ', tweet)

    # Replace #word with word
    # tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
    tweet = re.sub(r'#([^\s]+)', r'', tweet)

    tweet = re.sub(r'[^\x00-\x7F]', '', tweet)

    tweet = re.sub(r'\"', '', tweet)

    # trim
    tweet = tweet.strip()

    # remove last " at string end
    tweet = tweet.rstrip('\'"')
    tweet = tweet.lstrip('\'"')

    tweet = replaceThreeOrMore(tweet)

    return tweet


def replaceThreeOrMore(s):
    # pattern to look for three or more repetitions of any character, including
    # newlines.
    pattern = re.compile(r"(.)\1{1,}", re.DOTALL)

    return pattern.sub(r"\1\1", s)


def get_all_tweets(query, limit=1000):

    config = parse_config()

    # authorize twitter, initialize tweepy
    auth = tweepy.OAuthHandler(config.get('consumer_key'), config.get('consumer_secret'))
    auth.set_access_token(config.get('access_key'), config.get('access_secret'))
    api = tweepy.API(auth)

    # query users from the USA
    try:
        places = api.geo_search(query="USA", granularity="country")
    except tweepy.TweepError, e:
        if e.api_code == 215:
            print "Error using tweepy API, check config.json file for authentication/key information."
            sys.exit(1)
        else:
            print e
    place_id = places[0].id

    # initialize a list to hold all the tweepy Tweets
    alltweets = []

    # make initial request for most recent tweets (200 is the maximum allowed count)
    new_tweets = api.search(q="#%s -RT place:%s"%(query, place_id), count=200)

    # save most recent tweets
    alltweets.extend(new_tweets)

    # save the id of the oldest tweet less one
    oldest = alltweets[-1].id - 1

    processed_tweets = []
    # keep grabbing tweets until there are no tweets left to grab
    while len(processed_tweets) < limit:
        print "getting tweets before %s" % (oldest)

        # all subsequent requests use the max_id param to prevent duplicates
        new_tweets =api.search(q="#%s -RT place:%s"%(query, place_id), count=200, max_id=oldest)

        # save most recent tweets
        alltweets.extend(new_tweets)

        # update the id of the oldest tweet less one
        oldest = alltweets[-1].id - 1

        # clean and sanitize the data
        outtweets = [ process_tweet(tweet.text.encode("utf-8")) for tweet in alltweets]
        processed_tweets.extend( [x for x in outtweets if not 'AT_USER' in x and not 'URL' in x and len(x.split(" ")) > 3] )

        print "...%s tweets downloaded so far" % (len(processed_tweets))


    # write the csv
    with open('training_data/%s_tweets.txt' % query, 'wb') as f:
        writer = csv.writer(f)
        writer.writerow(["text"])
        for row in processed_tweets:
            writer.writerow([row])

    pass


if __name__ == '__main__':
    # pass in the hashtag to query for and the max number of messages to download
    parser = optparse.OptionParser()
    parser.add_option('-q', '--query',
        action="store", dest="query",
        help="hashtag query string", default="sad")

    parser.add_option('-l', '--limit', type="int",
        action="store", dest="limit",
        help="max limit of tweets to save", default=1000)

    options, args = parser.parse_args()

    get_all_tweets(options.query, options.limit)
