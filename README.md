Sentiment Analysis
======================

This is a tool for sentiment analysis of Facebook comments to classify comments as positive, negative, or neutral.
 There are also ipython notebooks available for visualizing and summarizing some of the data.

Requirements
------------
* nltk == 3.1
* tweepy == 3.5.0
* beautifulsoup4 == 4.4.1
* numpy == 1.11.0
* scikit_learn == 0.17.1

You may install the requirements by using pip:

    pip install -r requirements.txt

How to use
----------
1. Install dependencies
2. Download twitter results for training data
3. Generate the model
4. Run the classifier

Or run the following commands:

    ./download_tweets.py --query <myquery> --limit <limit>
    ./create_model.py --positive <path to positive training data file> --negative <path to neg training data file> \
        --neutral <path to neutral training data file> --classifier <LinearSVC|Tfid|Bernoulli|NaiveBayes>
    ./classifiy.py --file <path to file> --classifier <LinearSVC|Tfid|Bernoulli|NaiveBayes>


Training data
-------------

For the training data, I chose to use tweets with hashtag  "hope" for my positive data, "depressed" for negative data,
 and "fact" for neutral data. All tweets are processed and cleaned to remove as much noise as possible.

I use 10-fold cross validation and select the model with the best specificity.


References
----------
[1]: http://www.laurentluce.com/posts/twitter-sentiment-analysis-using-python-and-nltk/
[2]: https://github.com/victorneo/Twitter-Sentimental-Analysis
[3]: http://www.nltk.org/
