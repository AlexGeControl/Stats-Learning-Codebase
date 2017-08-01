# Data paths:
train_path = "../resource/lib/publicdata/aclImdb/train/" # use terminal to ls files under this directory
test_path = "../resource/lib/publicdata/imdb_te.csv" # test data for grade evaluation
test_filename = "../resource/asnlib/public/imdb_te.csv"

stopwords_filename = "stopwords.en.txt"

# Raw inputs:
TRAIN_RAW = 'imdb_tr.csv'
TEST_RAW = 'imdb_te.csv'
# Processed inputs:
TRAIN_FEATURES_PROCESSED = 'X_train.pkl'
TRAIN_LABELS_PROCESSED = 'y_train.pkl'
TEST_FEATURES_PROCESSED = 'X_test.pkl'
# Outputs:
UNIGRAM_OUTPUT = "unigram.output.txt"
BIGRAM_OUTPUT = "bigram.output.txt"
UNIGRAM_TFIDF_OUTPUT = "unigramtfidf.output.txt"
BIGRAM_TFIDF_OUTPUT = "bigramtfidf.output.txt"

from os.path import isfile

import re
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd

import string
from nltk.corpus import stopwords as sw
from nltk.corpus import wordnet as wn
from nltk import wordpunct_tokenize
from nltk import WordNetLemmatizer
from nltk import sent_tokenize
from nltk import pos_tag
from sklearn.base import BaseEstimator, TransformerMixin

import pickle

def identity(arg):
    """
    Simple identity function works as a passthrough.
    """
    return arg

class NLTKPreprocessor(BaseEstimator, TransformerMixin):
    """
    """
    def __init__(
        self, stopwords=None, punct=None, lower=True, strip=True
    ):
    	self.lower      = lower
    	self.strip      = strip
    	self.stopwords  = stopwords or set(sw.words('english'))
    	self.punct      = punct or set(string.punctuation)
    	self.lemmatizer = WordNetLemmatizer()

    def fit(self, X, y=None):
        return self

    def inverse_transform(self, X):
        return [" ".join(doc) for doc in X]

    def transform(self, X):
        return [
            list(self.tokenize(doc)) for doc in X
        ]

    def tokenize(self, document):
    	# Break the document into sentences
    	for sent in sent_tokenize(document):
    	    # Break the sentence into part of speech tagged tokens
    	    for token, tag in pos_tag(wordpunct_tokenize(sent)):
    	        # Apply preprocessing to the token
    	        token = token.lower() if self.lower else token
    	        token = token.strip() if self.strip else token
    	        # token = token.strip('_') if self.strip else token
    	        # token = token.strip('*') if self.strip else token

    	        # If stopword, ignore token and continue
    	        if token in self.stopwords:
    	            continue

    	        # If punctuation, ignore token and continue
    	        # if all(char in self.punct for char in token):
    	        #    continue

    	        # Lemmatize the token and yield
    	        lemma = self.lemmatize(token, tag)
    	        yield lemma

    def lemmatize(self, token, tag):
    	tag = {
    	    'N': wn.NOUN,
    	    'V': wn.VERB,
    	    'R': wn.ADV,
    	    'J': wn.ADJ
    	}.get(tag[0], wn.NOUN)

    	return self.lemmatizer.lemmatize(token, tag)

# Feature extractor:
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
# Pre-processor:
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler
# Cross validation:
from sklearn.model_selection import StratifiedShuffleSplit
# Classifier:
from sklearn.linear_model import SGDClassifier
# Evaluation metric:
from sklearn.metrics import accuracy_score
from sklearn.metrics import make_scorer
# Hyperparameter tuning:
from sklearn.model_selection import GridSearchCV

def imdb_data_preprocess(inpath, outpath="./", name="imdb_tr.csv", mix=False):
    '''
    Extract and combine text files under train_path directory into imdb_tr.csv.
    Each text file in train_path would be stored as a row in imdb_tr.csv.
    And imdb_tr.csv should have two columns, "text" and label'''
    # Set up session:
    from os.path import join
    from os import listdir

    # row number and rating parser:
    parser = re.compile(r"(\d+)_(\d+).txt")
    # Label mapper:
    label_map = {
        'pos': 1,
        'neg': 0
    }
    # Initialize dataframe dict:
    df_in_dict = {
        'row_number': [],
        'review_text': [],
        'polarity': []
    }

    # Parse records:
    for sentiment in label_map:
        data_dir_path = join(inpath, sentiment)
        for data_filename in listdir(data_dir_path):
            # Parse row number and rating
            fields = parser.match(data_filename)
            row_number, rating = int(fields.group(1)), int(fields.group(2))

            # Parse comment:
            with open(
                join(data_dir_path, data_filename)
            ) as data:
                review_text = data.read()

            # Add to dict:
            df_in_dict['row_number'].append(row_number)
            df_in_dict['review_text'].append(review_text)
            df_in_dict['polarity'].append(1 if sentiment == 'pos' else 0)

    # Convert to dataframe:
    df = pd.DataFrame.from_dict(df_in_dict)

    # Parallel processing:
    df['text'] = df['review_text'].apply(
        # 1. Remove HTML:
	    lambda x: BeautifulSoup(x, "html5lib").get_text()
    ).apply(
        # 2. Remove non-letters:
	    lambda x: re.sub("[^a-zA-Z]", " ", x)
    )

    # Format column names:
    df = df[
        ['row_number', 'text', 'polarity']
    ]
    df.columns = ['', 'text', 'polarity']

    # Save to disk:
    df.to_csv(
        join(outpath, name),
        index = False
    )

    return df

def show_most_informative_features(model, n=20):
    """
    Accepts a Pipeline with a classifer and a TfidfVectorizer and computes
    the n most informative features of the model. If text is given, then will
    compute the most informative features for classifying that text.
    Note that this function will only work on linear models with coefs_
    """
    # Set up session:
    from operator import itemgetter

    # Get vectorizer & classifier:
    vectorizer = model.named_steps['vect']
    classifier = model.named_steps['clf']

    # Use the coefficients
    tvec = classifier.coef_

    # Zip the feature names with the coefs and sort
    coefs = sorted(
        zip(tvec[0], vectorizer.get_feature_names()),
        key=itemgetter(0), reverse=True
    )

    topn  = zip(coefs[:n], coefs[:-(n+1):-1])

    # Create the output string to return
    output = []

    # Create two columns with most negative and most positive features.
    for (cp, fnp), (cn, fnn) in topn:
        output.append(
            "{:0.4f}{: >15}    {:0.4f}{: >15}".format(cp, fnp, cn, fnn)
        )

    print "\n".join(output)

def get_best_parameter(vectorizer, X_train, y_train):
    """ Get best parameter for SGD classifier using given tokenizer
    """
    # Create cross-validation sets from the training data
    cv_sets = StratifiedShuffleSplit(n_splits = 5, test_size = 0.30, random_state = 42).split(X_train, y_train)

    # Model:
    model = Pipeline([
        ('vect', vectorizer),
        ('clf', SGDClassifier(
            loss='hinge',
            penalty='l1',
            fit_intercept=True,
            shuffle=True,
            eta0 = 0.1,
            learning_rate='invscaling',
            random_state=42,
            n_jobs=4
        )),
    ])

    # Hyperparameters:
    params = {
        # 1. Max document frequency:
        "vect__max_df": (0.60, 0.75, 0.90),
        # 2. Max num. of features:
        "vect__max_features": (None, 5000, 10000),
        # 3. Initial learning rate:
        "clf__eta0": (0.1, 0.3, 0.5, 1.0),
        # 4. Regularization:
        "clf__alpha": (0.0, 0.000001, 0.00001, 0.0001),
        # 5. Num. of iterations:
        "clf__n_iter": (20, 40, 60)
    }

    # Make an fbeta_score scoring object
    scorer = make_scorer(accuracy_score)

    # Perform grid search on the classifier using 'scorer' as the scoring method
    grid_searcher = GridSearchCV(
        estimator = model,
        param_grid = params,
        scoring = scorer,
        cv = cv_sets,
        n_jobs = 2,
        verbose = 1
    )

    # Fit the grid search object to the training data and find the optimal parameters
    grid_fitted = grid_searcher.fit(X_train, y_train)

    # Get the estimator
    best_model = grid_fitted.best_estimator_

    # Get parameters & scores:
    best_parameters, score, _ = max(grid_fitted.grid_scores_, key=lambda x: x[1])

    print "[Best Parameters]:"
    print best_parameters
    print "[Best Score]:"
    print score

    return best_model

def train_and_predict(vectorizer, X_train, y_train, X_test, is_grid_search, **kwargs):
    """

    """
    if is_grid_search:
        # Find best regularization alpha:
        best_model = get_best_parameter(vectorizer, X_train, y_train)
        show_most_informative_features(best_model)
    else:
        # Vectorizer:
        vectorizer.set_params(
            max_df = kwargs['max_df'],
            max_features = kwargs['max_features']
        )

        # Model:
        model = Pipeline([
            ('vect', vectorizer),
            ('clf', SGDClassifier(
                loss='hinge',
                penalty='l1',
                fit_intercept=True,
                shuffle=True,
                learning_rate='invscaling',
                random_state=42,
                n_jobs=2,
                eta0 = kwargs['eta0'],
                alpha = kwargs['alpha'],
                n_iter = kwargs['n_iter']
            )),
        ])

        # Fit:
        model.fit(X_train, y_train)

        # Predict:
        y_pred = model.predict(X_test)

        return y_pred

if __name__ == "__main__":
    # Load stopwords:
    with open(stopwords_filename) as stop_words_file:
        stop_words = stop_words_file.read().splitlines()
    # Initialize NLTK preprocessor:
    preprocessor = NLTKPreprocessor(stopwords = stop_words)

    # Load training & testing datasets:
    print "[Process Training Set]: ..."
    if isfile(TRAIN_RAW):
        df_train = pd.read_csv(TRAIN_RAW)
    else:
        df_train = imdb_data_preprocess(train_path)
    print "[Process Training Set]: Done."

    print "[Process Testing Set]: ..."
    if isfile(TEST_RAW):
        df_test = pd.read_csv(TEST_RAW)
    else:
        df_test = pd.read_csv(test_filename, encoding = "ISO-8859-1")
        df_test['text'] = df_test['text'].apply(
            # 1. Remove HTML:
            lambda x: BeautifulSoup(x, "html5lib").get_text()
        ).apply(
            # 2. Remove non-letters:
            lambda x: re.sub("[^a-zA-Z]", " ", x)
        )
        df_test.to_csv(TEST_RAW, index=False)
    print "[Process Testing Set]: Done."

    print "[Lemmatization]: ..."
    if (
        isfile(TRAIN_FEATURES_PROCESSED) and
        isfile(TRAIN_LABELS_PROCESSED) and
        isfile(TEST_FEATURES_PROCESSED)
    ):
        with open(TRAIN_FEATURES_PROCESSED, 'rb') as X_train_pkl:
            X_train = pickle.load(X_train_pkl)
        with open(TRAIN_LABELS_PROCESSED, 'rb') as y_train_pkl:
            y_train = pickle.load(y_train_pkl)
        with open(TEST_FEATURES_PROCESSED, 'rb') as X_test_pkl:
            X_test = pickle.load(X_test_pkl)
    else:
        X_train = preprocessor.transform(df_train['text'])
        y_train = df_train['polarity'].values
        X_test = preprocessor.transform(df_test['text'])
        with open(TRAIN_FEATURES_PROCESSED, 'wb') as X_train_pkl:
            pickle.dump(X_train, X_train_pkl)
        with open(TRAIN_LABELS_PROCESSED, 'wb') as y_train_pkl:
            pickle.dump(y_train, y_train_pkl)
        with open(TEST_FEATURES_PROCESSED, 'wb') as X_test_pkl:
            pickle.dump(X_test, X_test_pkl)
    print "[Lemmatization]: Done"

    '''train a SGD classifier using unigram representation,
    predict sentiments on imdb_te.csv, and write output to
    unigram.output.txt'''
    print "[Unigram]:..."

    unigram_vectorizer = CountVectorizer(
        tokenizer=identity,
        preprocessor=None,
        lowercase=False,
        ngram_range = (1, 1),
        min_df = 15,
        dtype = np.float
    )
    '''
    y_pred = train_and_predict(unigram_vectorizer, X_train, y_train, X_test, is_grid_search = True)
    '''
    y_pred = train_and_predict(
        unigram_vectorizer,
        X_train, y_train, X_test, is_grid_search = False,
        max_df = 0.75,
        max_features = None,
        eta0 = 0.1,
        alpha = 1e-05,
        n_iter = 40
    )
    np.savetxt(UNIGRAM_OUTPUT, y_pred, delimiter=",", fmt="%d")

    print "[Unigram]: Done"

    '''train a SGD classifier using bigram representation,
    predict sentiments on imdb_te.csv, and write output to
    bigram.output.txt'''
    print "[Bigram]:..."

    bigram_vectorizer = CountVectorizer(
        tokenizer=identity,
        preprocessor=None,
        lowercase=False,
        ngram_range = (1, 2),
        min_df = 15,
        dtype = np.float
    )
    '''
    y_pred = train_and_predict(bigram_vectorizer, X_train, y_train, X_test, is_grid_search = True)
    '''
    y_pred = train_and_predict(
        bigram_vectorizer,
        X_train, y_train, X_test, is_grid_search = False,
        max_df = 0.75,
        max_features = None,
        eta0 = 0.1,
        alpha = 1e-6,
        n_iter = 60
    )
    np.savetxt(BIGRAM_OUTPUT, y_pred, delimiter=",", fmt="%d")

    print "[Bigram]: Done"

    '''train a SGD classifier using unigram representation
    with tf-idf, predict sentiments on imdb_te.csv, and write
    output to unigramtfidf.output.txt'''
    print "[Unigram with TFIDF]:..."

    unigram_tfidf_vectorizer = TfidfVectorizer(
        tokenizer=identity,
        preprocessor=None,
        lowercase=False,
        ngram_range = (1, 1),
        min_df = 15,
        use_idf = True,
        smooth_idf = False,
        sublinear_tf = True,
        dtype = np.float
    )
    '''
    y_pred = train_and_predict(unigram_tfidf_vectorizer, X_train, y_train, X_test, is_grid_search = True)
    '''
    y_pred = train_and_predict(
        unigram_tfidf_vectorizer,
        X_train, y_train, X_test, is_grid_search = False,
        max_df = 0.6,
        max_features = None,
        eta0 = 1.0,
        alpha = 0.0,
        n_iter = 60
    )
    np.savetxt(UNIGRAM_TFIDF_OUTPUT, y_pred, delimiter=",", fmt="%d")

    print "[Unigram with TFIDF]: Done"

    '''train a SGD classifier using bigram representation
    with tf-idf, predict sentiments on imdb_te.csv, and write
    output to bigramtfidf.output.txt'''
    print "[Bigram with TFIDF]:..."

    bigram_tfidf_vectorizer = TfidfVectorizer(
        tokenizer=identity,
        preprocessor=None,
        lowercase=False,
        ngram_range = (1, 2),
        min_df = 15,
        use_idf = True,
        smooth_idf = False,
        sublinear_tf = True,
        dtype = np.float
    )
    '''
    y_pred = train_and_predict(bigram_tfidf_vectorizer, X_train, y_train, X_test, is_grid_search = True)
    '''
    y_pred = train_and_predict(
        bigram_tfidf_vectorizer,
        X_train, y_train, X_test, is_grid_search = False,
        max_df = 0.6,
        max_features = 10000,
        eta0 = 1.0,
        alpha = 0.0,
        n_iter = 60
    )
    np.savetxt(BIGRAM_TFIDF_OUTPUT, y_pred, delimiter=",", fmt="%d")

    print "[Bigram with TFIDF]: Done"
