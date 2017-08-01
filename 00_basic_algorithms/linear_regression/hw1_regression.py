
# coding: utf-8

# # ColumbiaX--Assignment-01--Linear-Regression

# ## Set Up Session
import sys
import numpy as np

if __name__ == "__main__":
    # Precondition:
    if len(sys.argv) != 6:
        print "usage: python hw1_regression.py lambda sigma2 X_train.csv y_train.csv X_test.csv"
        exit(1)

    ## Parse parameters:
    # a. lambda for weight generation--non-negative integer:
    w_lambda = int(sys.argv[1])
    # b. sigma squared for observation--arbitrary positive number:
    y_sigma2 = int(sys.argv[2])
    def parse_features(filename):
        with open(filename) as f:
            lines = f.readlines()
        features = np.asarray(
            [[float(val) for val in line.split(",")] for line in lines]
        )
        return features
    # c. 'X_train.csv'
    X_train = parse_features(sys.argv[3])
    # d. 'y_train.csv'
    y_train = parse_features(sys.argv[4])
    # e. 'X_test.csv'
    X_test = parse_features(sys.argv[5])

    # ## Posterior Estimation
    #
    # Posterior covariance matrix:
    w_cov_posterior = np.linalg.pinv(w_lambda + np.matmul(X_train.T, X_train) / y_sigma2)


    # ## Ridge Regression
    #
    # Given samples and observations, solve the following ridge regression problem under the given generation & observation noises
    #
    # $$
    # w_{RR} = \arg\min_w \|y - Xw\|^2 + \lambda\|w\|^2.
    # $$
    #
    # Which is
    #
    # $$
    # w_{RR} = (\lambda\sigma^2I + X^TX)^{-1}X^Ty
    # $$
    #
    w_rr = np.matmul(
        #w_cov_posterior / y_sigma2,
        np.linalg.pinv(w_lambda + np.matmul(X_train.T, X_train)),
        np.matmul(
            X_train.T,
            y_train
        )
    ).ravel()

    # ## Active Learning
    #
    # Posterior estimation variance:
    y_sigma2_posterior = np.asarray(
        [
            np.matmul(
                x,
                np.matmul(
                    w_cov_posterior,
                    x.T
                )
            )
            for x in X_test
        ]
    )
    # Select top 10:
    probe_sequence = 1 + np.argsort(y_sigma2_posterior)[::-1][:10]


    # ## Generate Output
    #
    # Ridge regression weights:
    w_rr_output_name = "wRR_{w_lambda}.csv".format(w_lambda=w_lambda)
    with open(w_rr_output_name, "w") as w_rr_output:
        w_rr_output.write("%s" % "\n".join(str(w_rr_val) for w_rr_val in w_rr))

    # Active learning probe sequence:
    probe_seq_output_name = "active_{w_lambda}_{y_sigma2}.csv".format(w_lambda=w_lambda, y_sigma2=y_sigma2)
    with open(probe_seq_output_name, "w") as probe_seq_output:
        probe_seq_output.write("%s" % ",".join(str(probe_sequence_idx) for probe_sequence_idx in probe_sequence))

    exit(0)
