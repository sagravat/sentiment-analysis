import nltk
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
import re
import numpy as np

"""
Sanjay Agravat

    Classification utility methods for cleaning text, reading training data, classification, accuracy,
    and feature extraction

"""


def clean_text(raw_text, remove_stopwords = False, output_format ="string"):
    """
    Input:
            raw_text: raw text from input
            remove_stopwords: a boolean variable to indicate whether to remove stop words
            output_format: if "string", return a cleaned string
                           if "list", a list of words extracted from cleaned string.
    Output:
            Cleaned string or list.
    """

    # Remove HTML markup
    text = BeautifulSoup(raw_text, "lxml")

    # Keep only characters
    text = re.sub("[^a-zA-Z]", " ", text.get_text())

    # Split words and store to list
    text = text.lower().split()

    if remove_stopwords:

        # Use set as it has O(1) lookup time
        stops = set(stopwords.words("english"))
        words = [w for w in text if w not in stops]

    else:
        words = text

    # Return a cleaned string or list
    if output_format == "string":
        return " ".join(words)

    elif output_format == "list":
        return words


def get_words_in_tweets(tweets):
    all_words = []
    for (words, sentiment) in tweets:
      all_words.extend(words)
    return all_words


def get_word_features(wordlist, max=0):
    wordlist = nltk.FreqDist(wordlist)
    word_features = [x[0] for x in wordlist.most_common(max)]
    return word_features


def read_tweets(fname, t_type):
    tweets = []
    f = open(fname, 'r')
    line = f.readline()
    while line != '':
        tweets.append([line, t_type])
        line = f.readline()
    f.close()
    return tweets


def extract_features(document):
    word_features = get_word_features( document, 2000 )

    document_words = set(document)
    features = {}
    for word in word_features:
      features['contains(%s)' % word] = (word in document_words)
    return features


def classify_unknown(classifier, tweet):
    return classifier.classify(extract_features(nltk.word_tokenize(tweet)))


def classify_tweet(classifier, tweet):
    return classifier.classify(extract_features(tweet))


def classify_many_tweets(classifier, tweets):
    return classifier.classify_many([ extract_features(x) for x in tweets])


def accuracy(classifier, gold):
    l_pos = np.array(classify_many_tweets(classifier,  [x[0] for x in gold if x[1] == 'positive'] ))
    l_neg = np.array(classify_many_tweets(classifier,  [x[0] for x in gold if x[1] == 'negative'] ))
    l_neu = np.array(classify_many_tweets(classifier,  [x[0] for x in gold if x[1] == 'neutral']  ))

    print "\nConfusion matrix:\n%d\t%d\t%d\n%d\t%d\t%d\n%d\t%d\t%d" % (
          (l_pos == 'positive').sum(), (l_pos == 'negative').sum(), (l_pos == 'neutral').sum(),
          (l_neg == 'positive').sum(), (l_neg == 'negative').sum(), (l_neg == 'neutral').sum(),
          (l_neu == 'positive').sum(), (l_neu == 'negative').sum(), (l_neu == 'neutral').sum())

    total_neg = (l_pos == 'negative').sum() + (l_neg == 'negative').sum() + (l_neu == 'negative').sum()
    specificity = (float((l_neg == 'negative').sum()) / float(total_neg)) * 100.0
    total = correct = len(gold)
    for tweet in gold:
        if classify_tweet(classifier, tweet[0]) != tweet[1]:
            correct -= 1

    total_accuracy = (float(correct) / float(total))*100
    print "total accuracy ", total_accuracy
    print "Specificity ", specificity
    return (total_accuracy, specificity)