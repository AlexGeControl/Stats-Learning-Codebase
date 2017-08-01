
# coding: utf-8

# # ColumbiaX-04-Probabilistic-Matrix-Factorization

# ## Set Up Session
import sys
import numpy as np

# Config:
d = 5
o_sigma2 = 0.1
h_lambda = 2
# Maximum number of iterations:
MAX_ITER = 50
# Monitor epochs:
MONITOR_EPOCHES = set([10, 25, 50])

# Main:
if __name__ == "__main__":
    # Precondition:
    if len(sys.argv) != 2:
        print "usage: python hw4_PMF.py ratings.csv"
        exit(1)

    ## Parse ratings:
    def parse_ratings(filename):
        with open(filename) as f:
            lines = f.readlines()
        ratings = np.asarray(
            [[float(val) for val in line.split(",")] for line in lines]
        )
        return ratings
    # 'ratings.csv'
    X = parse_ratings(sys.argv[1]) - np.asarray([1, 1, 0.0])

    # Parse user & object index:
    pmf_object_mapping, pmf_user_mapping = {}, {}
    pmf_rating_mapping = {}
    for pmf_user_index, pmf_object_index, pmf_rating in zip(
        X[:, 0].astype(np.int),
        X[:, 1].astype(np.int),
        X[:, 2]
    ):
        # Update object mapping:
        object_list = pmf_object_mapping.get(pmf_user_index, [])
        object_list.append(pmf_object_index)
        pmf_object_mapping[pmf_user_index] = object_list
        # Update user mapping:
        user_list = pmf_user_mapping.get(pmf_object_index, [])
        user_list.append(pmf_user_index)
        pmf_user_mapping[pmf_object_index] = user_list
        # Update pmf_rating:
        pmf_rating_mapping[(pmf_user_index, pmf_object_index)] = pmf_rating

    # Get the number of users & objects:
    pmf_N_u, pmf_N_v = max(pmf_object_mapping.keys())+1, max(pmf_user_mapping.keys())+1

    # ## Probabilistic Matrix Factorization
    #
    # $$
    # \mathcal{L} = -\sum_{(i,j)\in\Omega} \frac{1}{2\sigma^2}(M_{ij} - u_i^Tv_j)^2  - \sum_{i=1}^{N_u}\frac{\lambda}{2}\|u_i\|^2 - \sum_{j=1}^{N_v}\frac{\lambda}{2}\|v_j\|^2
    # $$

    # Initialize:
    pmf_U = np.zeros((pmf_N_u, d))
    pmf_h_mu = np.zeros(d)
    pmf_h_cov = 1.0 / h_lambda * np.eye(d)
    pmf_V = np.random.multivariate_normal(mean=pmf_h_mu, cov=pmf_h_cov, size = pmf_N_v)

    # Loss function monitor:
    pmf_loss = []

    # Compute:
    for epoch in xrange(MAX_ITER):
        # Update user matrix:
        for pmf_user_index in pmf_object_mapping.keys():
            pmf_object_list = pmf_object_mapping[pmf_user_index]
            pmf_U[pmf_user_index] = np.matmul(
                np.linalg.pinv(
                    np.matmul(pmf_V[pmf_object_list].T, pmf_V[pmf_object_list]) + h_lambda * o_sigma2
                ),
                np.matmul(
                    pmf_V[pmf_object_list].T,
                    np.asarray([pmf_rating_mapping[(pmf_user_index, o)] for o in pmf_object_list])
                )
            )
        # Update user matrix:
        for pmf_object_index in pmf_user_mapping.keys():
            pmf_user_list = pmf_user_mapping[pmf_object_index]
            pmf_V[pmf_object_index] = np.matmul(
                np.linalg.pinv(
                    np.matmul(pmf_U[pmf_user_list].T, pmf_U[pmf_user_list]) + h_lambda * o_sigma2
                ),
                np.matmul(
                    pmf_U[pmf_user_list].T,
                    np.asarray([pmf_rating_mapping[(u, pmf_object_index)] for u in pmf_user_list])
                )
            )

        # Calculate loss:
        pmf_cur_loss = 0.0
        for pmf_index, pmf_rating in pmf_rating_mapping.iteritems():
            pmf_user_index, pmf_object_index = pmf_index
            pmf_cur_loss += 1.0 / (2*o_sigma2)*(pmf_rating - np.dot(pmf_U[pmf_user_index], pmf_V[pmf_object_index]))**2
        pmf_cur_loss += h_lambda/2 * (np.sum(pmf_U**2)+np.sum(pmf_V**2))
        pmf_loss.append(-pmf_cur_loss)

        # Generate output:
        if epoch+1 in MONITOR_EPOCHES:
            # User matrix:
            with open(
                "U-{iteration}.csv".format(iteration = epoch+1),
                "w"
            ) as U_output:
                U_output.write(
                    "\n".join(
                        ",".join(str(x) for x in xs) for xs in pmf_U
                    )
                )
            # Object matrix:
            with open(
                "V-{iteration}.csv".format(iteration = epoch+1),
                "w"
            ) as V_output:
                V_output.write(
                    "\n".join(
                        ",".join(str(x) for x in xs) for xs in pmf_V
                    )
                )
    # Finally:
    with open("objective.csv","w") as objective_output:
        objective_output.write(
            "\n".join(str(x) for x in pmf_loss)
        )
    '''
    # Evaluate:
    pmf_O = np.matmul(
        pmf_U,
        pmf_V.T
    )
    for pmf_user_index, pmf_object_index, pmf_rating in zip(
        X[:, 0].astype(np.int),
        X[:, 1].astype(np.int),
        X[:, 2]
    ):
        print "[PMF Error]: {:.2f}".format(
            np.abs(pmf_O[pmf_user_index, pmf_object_index] - pmf_rating)
        )
    '''

    exit(0)
