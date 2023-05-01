import numpy as np
from sklearn import gaussian_process
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from datetime import datetime
from numpy.random import RandomState
from scipy.linalg import cholesky, cho_solve
from operator import itemgetter
import matplotlib.pyplot as plt
import pickle

from utils_cbo import x_data, checker_partitions, gen_cls_dataset, gen_y_dataset_har, gen_idx_thr, bo_par, \
    bo_hetero, bo_test, leaf_node_data, propose_location, propose_location_gp, ucb_sampling, gen_y_levy, \
    gen_y_qing, gen_y_rosenbrock, gen_y_levy_cen, gen_y_qing_cen, gen_y_rosenbrock_cen, checker_part_unbalanced, \
    objective_func
from plot_cbo import plot_err_bars, plot_rewards

from time import time
import warnings

warnings.filterwarnings('ignore')

# Initializations
gradient_eval = True  # Evaluation of gradient of the function for minimization
GPR_CHOLESKY_LOWER = True  # Cholesky solve to compute the inverse in computation of posterior


def bo_multi_test_che(total_budget, dimension, tup, k, n_partitions, bounds, test_data_size,
                      kernel, file_name, theta_initial, max_depth, w, beta, c, runs=1, alpha=0.1,
                      min_samp=2, restarts=1, normalize=True, weighted=True, std=0):
    """bo_multi_test is the function to run multiple tests across different bo_functions with same initial dataset
    for all functions in a run"""

    """Initialization"""
    class_bo_mse = []
    bo_par_mse = []
    bo_gp_mse = []
    par_mse_par = []
    cbo_mse_par = []
    freq = w
    data = {}

    """Checker Pattern Partitions and clf_init for all the Runs"""
    par, label_mat = checker_partitions(tup, dimension, k, bounds)
    par2 = [par[j][1:-1] for j in range(dimension)]
    partition_mat = np.arange(n_partitions).reshape(tup)
    print(par, '\n', label_mat, '\n', partition_mat, '\n')

    """Looping through the number of runs"""
    for j in range(runs):
        print(j)  # Run Number
        x_dataset = x_data(dimension, bounds, total_budget)  # Generating Uniformly Random input data (X)
        c_dataset = gen_cls_dataset(x_dataset, par, label_mat)  # cls labels for the generated data
        p_dataset = gen_cls_dataset(x_dataset, par, partition_mat)  # partition labels for the generated data
        y_dataset = gen_y_dataset_har(x_dataset, c_dataset, p_dataset, w, beta, c, alpha=alpha,
                                      std=std)  # Output y_dataset
        # y_dataset = gen_y_qing_cen(x_dataset, c_dataset, p_dataset, par, partition_mat, w, beta, c, alpha)
        # freq += [w[0]]
        # beta_mat += [beta]
        """Uniform Split and this remains same for each set of experiments with same random seed, dimension and 
        budget"""
        x_train, x_test, c_train, c_test, y_train, y_test = train_test_split(x_dataset, c_dataset, y_dataset,
                                                                             test_size=test_data_size)
        idx_train, thr_train = gen_idx_thr(x_train, par2)

        """BO Partition - Individual update in each partition"""
        # print("Partition BO")
        ker = kernel
        clf_par, leaf_nodes_par, leaf_cls_par, leaf_node_data_par, gps_par \
            = bo_par(x_train, y_train, c_train, idx_train, thr_train, ker, bounds, max_depth, min_samp=min_samp,
                     restarts=restarts, normalize=normalize)

        mse_par_par, n_par_par = bo_test(x_test, y_test, clf_par, leaf_nodes_par, gps_par)
        mse_par = sum([mse_par_par[ii] * n_par_par[ii] for ii in range(len(gps_par))]) / sum(n_par_par)
        bo_par_mse += [mse_par]
        par_mse_par += [mse_par_par.copy(), n_par_par.copy()]

        """Class BO - New Likelihood """
        # print("Class BO")
        clf_cbo, leaf_nodes_cbo, leaf_cls_cbo, leaf_node_data_cbo, gps_cbo \
            = bo_hetero(x_train, y_train, c_train, idx_train, thr_train, ker, bounds, theta_initial, max_depth,
                        min_samp=min_samp, restarts=restarts, normalize=normalize, weighted=weighted)

        mse_cbo_par, n_cbo_par = bo_test(x_test, y_test, clf_cbo, leaf_nodes_cbo, gps_cbo)
        mse_cbo = sum([mse_cbo_par[ii] * n_cbo_par[ii] for ii in range(len(gps_cbo))]) / sum(n_cbo_par)
        class_bo_mse += [mse_cbo]
        cbo_mse_par += [mse_cbo_par.copy(), n_cbo_par.copy()]

        """GP"""
        # print("Standard BO")
        bo_gp = gaussian_process.GaussianProcessRegressor(
            clone(ker), normalize_y=normalize, n_restarts_optimizer=restarts)
        bo_gp.fit(x_train, y_train)
        y_pred = bo_gp.predict(x_test)
        mse_gp = mean_squared_error(y_pred, y_test)
        bo_gp_mse += [mse_gp]

        """Data"""
        round_data = {"x_train": x_train, "x_test": x_test, "y_train": y_train, "y_test": y_test,
                      "label_mat": label_mat, "partitions": par, "leaf_cls_par": leaf_cls_par,
                      "leaf_cls_cbo": leaf_cls_cbo, "clf_par": clf_par, "clf_cbo": clf_cbo,
                      "gps_par": gps_par, "gps_cbo": gps_cbo, "bo_gp": bo_gp}
        data[j] = round_data

    mse_array = np.array([class_bo_mse, bo_par_mse, bo_gp_mse])  # Each list forms a row of matrix
    mse_df = pd.DataFrame(mse_array.T)  # Rows to columns
    mse_df.columns = ['Class BO', 'Partition BO', 'Standard BO']
    freq_df = pd.DataFrame(freq)
    par_mse_df = pd.DataFrame(par_mse_par).T
    par_mse_df.columns = ['par_par_mse', 'n_par'] * runs
    cbo_mse_df = pd.DataFrame(cbo_mse_par).T
    cbo_mse_df.columns = ['cbo_par_mse', 'n_par'] * runs
    mse_df = pd.concat([mse_df, freq_df, par_mse_df, cbo_mse_df], axis=1)
    excel_file_name = file_name + ".xlsx"
    mse_df.to_excel(excel_file_name)
    # data_file_name = file_name + ".pkl"
    # pickle_out = open(data_file_name, "wb")
    # pickle.dump(data, pickle_out)
    # pickle_out.close()

    return mse_df, data



if __name__ == '__main__':
    ''' Multiple runs of modeling error computation for'''

    for k in [2,4,6,8]:
        for total_budget in [5050, 5500, 6000]:
            for tups in [(2,), (4,), (6,), (8,)]:
                rstate = 175  # Random State
                dt = datetime.now().strftime("%m%d%y%H%M%S")
                # total_budget = 6000  # number of data points exp - 1000, 2000, 5000
                # k = 8  # Number of classes - exp - 2, 4, 8
                dimension = 2  # dimension -  exp - 2, 4, 8
                bounds = [(-10, 10)]*dimension  # bounds on the input data - same as space_bounds
                tup = tups * dimension
                n_partitions = tup[0]**dimension  # Number of partitions - exp - 20, 40, 80
                max_depth = 20  # Max depth of the tree
                test_data_size = 5000 / total_budget
                runs = 50  # Number of Macro reps
                kernel_def = ConstantKernel(1) * RBF(np.ones((dimension,))) + WhiteKernel()  # kernel for learning
                theta_initial = np.zeros_like(kernel_def.theta)  # initialization of log hyperparameters
                n_restarts = 3  # No of restarts of minimize optimizer
                initial_pt = None  # Initial point for minimize optimizer
                restarts = 1  # No of restarts of gp - theta initial is uniform random
                alpha = 1  # Multiple for beta in UCB (exploration)
                num_iter_refit = 10  # Refit the tree after this number of iterations
                min_samp = 2  # minimum number of leaf samples
                normalize = True  # Normalization of y while training GP's
                weighted = False # Likelihood is weighted with samples per partition

                """Vectors for Data_gen Function"""
                prng = RandomState(rstate)
                het_ratio = 1  # factor Gap between 2 set of frequencies
                w1 = np.arange(1, het_ratio * k + 1, het_ratio).reshape(-1, 1) *\
                     prng.randn(1, dimension)  # Fixed Frequency vector
                beta = (prng.randn(k, dimension))  # Fixed vector
                alpha = 0.1  # exponential factor
                c = prng.randn(n_partitions, 1)  # Fixed Offset/Intercept

                """ File name and Main Function """
                file_name = "Experiments/noisy_levy_cen_mse_" + str(dimension) + "d_" + str(total_budget) + "N_" \
                            + str(k) + "c_" + str(n_partitions) + "p_rbf_test_" + str(int(test_data_size * 100)) \
                            + "_bounds" + str(bounds[0]) + "_" + str(bounds[1]) + "_" + str(runs) + "runs_" \
                            + str(het_ratio) + "het_" + str(alpha) + "alp_" + dt
                print(file_name)
                np.random.seed(rstate)
                try:
                    mse_df, data = bo_multi_test_che(total_budget, dimension, tup, k, n_partitions, bounds,
                                                     test_data_size, kernel_def, file_name, theta_initial, max_depth,
                                                     w1, beta, c, runs, alpha=alpha, min_samp=min_samp,
                                                     restarts=restarts, normalize=normalize, weighted=weighted)
                except Exception as e:
                    print (k, total_budget, tups)
                    print(file_name)
                    print(f"Unexpected {e=}, {type(e)=}")
                    continue
                df = mse_df.iloc[:, 0:3]
                plt_df = df.dropna()
                plot_err_bars(plt_df, k, n_partitions, het_ratio, file_name)
                plt.show()

    ''' Modeling one run '''

    # rstate = 175  # Random State
    # dt = datetime.now().strftime("%m%d%y%H%M%S")
    # total_budget = 5500  # number of data points exp - 1000, 2000, 5000
    # dimension = 2  # dimension -  exp - 2, 4, 8
    # bounds = [(-10, 10)]*dimension  # bounds on the input data - same as space_bounds
    # k = 8  # Number of classes - exp - 2, 4, 8
    # tup = (8,)*dimension
    # n_partitions = tup[0] ** dimension  # Number of partitions - exp - 20, 40, 80
    # max_depth = 20  # Max depth of the tree
    # test_data_size = 5000 / total_budget
    # runs = 10  # Number of Macro reps
    # kernel_def = ConstantKernel(1) * RBF(np.ones((dimension,))) + WhiteKernel()  # kernel for learning
    # theta_initial = np.zeros_like(kernel_def.theta)  # initialization of log hyperparameters
    # # max_leaf_nodes = n_partitions  # 2*n_partitions or None   # max leaf nodes
    # # data_kernel = ConstantKernel(1) * RBF(np.ones((dimension,)))  # Kernel for data generation

    # """Vectors for Data_gen Function"""
    # prng = RandomState(rstate)
    # het_ratio = 1  # factor gap between 2 set of frequencies
    # w1 = np.arange(1, het_ratio * k + 1, het_ratio).reshape(-1, 1) *\
    #      prng.randn(1, dimension)  # Fixed Frequency vector
    # alpha = 0.1  # exponential factor
    # beta = (prng.randn(k, dimension))  # Fixed vector
    # c = prng.randn(n_partitions, 1)  # Fixed Offset/Intercept

    # """ File name and Main Function """
    # file_name = "Experiments/har_mse_" + str(dimension) + "d_" + str(total_budget) + "N_" \
    #             + str(k) + "c_" + str(n_partitions) + "p_rbf_test_" + str(int(test_data_size * 100)) \
    #             + "_bounds" + str(bounds[0]) + "_" + str(bounds[1]) + "_" + str(runs) + "runs_" \
    #             + str(het_ratio) + "het_" + str(alpha) + "alp_" + dt
    # print(file_name)
    # np.random.seed(rstate)
    # # try:
    # mse_df, data = bo_multi_test_che(total_budget, dimension, tup, k, n_partitions, bounds, test_data_size,
    #                                  kernel_def, file_name, theta_initial, max_depth, w1, beta, c, runs, alpha)
    # # except:
    # #     print (k, total_budget, tups)
    # #     print(file_name)
    # #     continue
    # df = mse_df.iloc[:, 0:3]
    # plt_df = df.dropna()
    # plot_err_bars(plt_df, k, n_partitions, hetero_ratio, file_name)

  
