#!/usr/bin/env python

"""

Sanjay Agravat

    Select a classifier to analyze for sentiment analysis on an input file with fixed format
    and output the results to a file.

"""

import csv
import sys
from sklearn.externals import joblib
import util
import optparse


# Run a classifier on an input file for sentiment analysis using a previously trained
# model and save the results to disk.
def classify(input_file, delimiter=",", classifier_param="LinearSVC"):
    data = []
    with open(input_file, 'rU') as f:
        reader = csv.reader(f, delimiter=delimiter)
        next(reader, None)  # skip the headers
        for row in reader:
            person = row[2]
            text = row[3]
            likes = row[5]
            data.append((text, person, likes))

    try:
        classifier = joblib.load("model/%s_classifier.pkl" % classifier_param)
    except IOError:
        print "unable to load classifier: %s. Exiting program." % classifier_param
        sys.exit(1)

    with open("results/%s_%s" %(classifier_param, input_file.replace("/","-")), 'wb') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t', quotechar='\"', quoting=csv.QUOTE_ALL)
        writer.writerow(["label", "message", "author", "likes"])

        for message, author, likes in data:
            cleaned_message = util.clean_text(message, True)
            if len(cleaned_message.split(" ")) > 3:
                writer.writerow([util.classify_unknown(classifier, cleaned_message), message, author, likes])

if __name__ == "__main__":

    DELIMITER = ","

    parser = optparse.OptionParser()
    parser.add_option('-f', '--file',
        action="store", dest="file",
        help="file to classify", default="notebooks/ucb_comments.csv")

    parser.add_option('-c', '--classifier',
        action="store", dest="classifier",
        help="classifier", default="LinearSVC")

    options, args = parser.parse_args()

    classify(options.file, DELIMITER, options.classifier)
