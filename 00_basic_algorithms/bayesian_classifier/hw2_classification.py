
# coding: utf-8

# # ColumbiaX-02-Bayesian-Classifier

# ## Set Up Session
import sys
import numpy as np
from scipy.stats import multivariate_normal

if __name__ == "__main__":
    # Precondition:
    if len(sys.argv) != 4:
        print "usage: python hw2_classification.py X_train.csv y_train.csv X_test.csv"
        exit(1)

    ## Parse parameters:
    def parse_features(filename):
        with open(filename) as f:
            lines = f.readlines()
        features = np.asarray(
            [[float(val) for val in line.split(",")] for line in lines]
        )
        return features
    # a. 'X_train.csv'
    X_train = parse_features(sys.argv[1])
    # b. 'y_train.csv'
    y_train = parse_features(sys.argv[2]).astype(np.int).ravel()
    # c. 'X_test.csv'
    X_test = parse_features(sys.argv[3])

    # ## Maximum-Likelihood Estimation for Priors & Generative Parameters
    # Class priors:
    prior_ml = np.bincount(y_train).astype(np.float) / y_train.shape[0]

    # Gaussian parameters:
    class_indices = np.unique(y_train)
    mu_ml = np.asarray(
        [np.mean(X_train[y_train == class_idx], axis = 0) for class_idx in class_indices]
    )

    cov_ml = np.asarray(
        [np.cov(X_train[y_train == class_idx].T, bias=True) for class_idx in class_indices]
    )

    # Posterior probabilities for test cases:
    pdfs = [multivariate_normal(mean=mu_ml[class_idx], cov=cov_ml[class_idx]) for class_idx in class_indices]
    posteriors = np.column_stack(
        tuple(
            prior_ml[class_idx]*pdfs[class_idx].pdf(X_test) for class_idx in class_indices
        )
    )
    posteriors = np.matmul(
        np.diag(1.0 / posteriors.sum(axis=1)),
        posteriors
    )

    # Predict using Bayesian classifier:
    y_pred = np.argmax(posteriors, axis = 1)
    # print "[Prediction Accuracy]: {:.2f}%".format(100.0*np.mean(y_pred == y_test))


    # ## Posterior probabilities for test instances:
    #
    with open("probs_test.csv", "w") as posteriors_output:
        posteriors_output.write(
            "\n".join(
                ",".join(str(val) for val in vals) for vals in posteriors
            )
        )
