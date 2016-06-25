#!/usr/bin/env python

"""

Sanjay Agravat
    Sentiment Analysis of Facebook comments using twitter data.
        - classify as positive, negative, or neutral

"""

import nltk
from nltk.classify.naivebayes import NaiveBayesClassifier
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from sklearn import cross_validation
from random import shuffle
import util
import optparse
import sys

# max number of features to used in the model
MAX_FEATURES = 2000


def extract_features(document):
    # extract the word features out from the training training_data
    word_features = util.get_word_features( document, MAX_FEATURES )

    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains(%s)' % word] = (word in document_words)
    return features



# Create a model from positive, negative, and neutral training training_data
# using a user-specified classifier parameter
# Performs k-fold cross validation and saves the best model to disk based
# using the model with the best specificity
def create_model(pos_tweets, neg_tweets, neu_tweets, classifier_param='LinearSVC'):

    # filter away words that are less than 3 letters to form the training training_data
    tweets = []
    for (words, sentiment) in pos_tweets + neg_tweets + neu_tweets:
        words = util.clean_text(words, True)
        words_filtered = [e.lower() for e in words.split() if len(e) >= 3]
        #words_filtered = [' '.join(w) for w in [ x for x in nltk.bigrams(words.split())]]
        tweets.append((words_filtered, sentiment))

    # make sure tweets are shuffled randomly
    shuffle(tweets)

    # get the training set and train the Classifier
    training_set = nltk.classify.util.apply_features(extract_features, tweets)

    max_specificity = -1
    best_classifier = None
    average_accuracy = 0.0

    # perform 10-fold cross validation
    cv = cross_validation.KFold(len(training_set), n_folds=10, shuffle=False, random_state=None)
    for traincv, testcv in cv:

        if classifier_param == "LinearSVC":
            classifier = SklearnClassifier(LinearSVC()).train(training_set[traincv[0]:traincv[len(traincv)-1]])
        elif classifier_param == "Tfid":
            # does TF-IDF weighting,
            # chooses the 1000 best features based on a chi2 statistic,
            # and then passes that into a multinomial naive Bayes classifier.
            pipeline = Pipeline([('tfidf', TfidfTransformer()), \
                                   ('chi2', SelectKBest(chi2, k=1000)), \
                                   ('nb', MultinomialNB())])
            classifier = SklearnClassifier(pipeline).train(training_set[traincv[0]:traincv[len(traincv)-1]])
        elif classifier_param == "Bernoulli":
            classifier = SklearnClassifier(BernoulliNB()).train(training_set[traincv[0]:traincv[len(traincv)-1]])
        elif classifier_param == "NaiveBayes":
            classifier = NaiveBayesClassifier.train(training_set[traincv[0]:traincv[len(traincv)-1]])
        else:
            print "Classifier option not available: ", classifier_param
            sys.exit(1)

        accuracy_of_classifier, specificity = \
            util.accuracy(classifier, tweets[testcv[0]:testcv[len(testcv)-1]])

        average_accuracy += accuracy_of_classifier
        if specificity > max_specificity:
            max_specificity = specificity
            best_classifier = classifier

    print "\naverage accuracy: ", average_accuracy/cv.n_folds

    # save the classifier
    joblib.dump(best_classifier, "model/%s_classifier.pkl" % classifier_param)

    print "saved classifier"

if __name__ == "__main__":

    print "read training training_data"
    # read in positive, negative, and neutral training tweets

    parser = optparse.OptionParser()
    parser.add_option('-p', '--positive',
        action="store", dest="positive",
        help="positive training data", default="training_data/hope_tweets_test.txt")

    parser.add_option('-n', '--negative',
        action="store", dest="negative",
        help="negative training data", default="training_data/depressed_tweets_test.txt")

    parser.add_option('-t', '--neutral',
        action="store", dest="neutral",
        help="neutral training data", default="training_data/fact_tweets_test.txt")

    parser.add_option('-c', '--classifier',
        action="store", dest="classifier",
        help="classifier", default="LinearSVC")


    options, args = parser.parse_args()

    pos_tweets = util.read_tweets(options.positive, 'positive')
    neg_tweets = util.read_tweets(options.negative, 'negative')
    neu_tweets = util.read_tweets(options.neutral, 'neutral')

    create_model(pos_tweets, neg_tweets, neu_tweets, options.classifier)