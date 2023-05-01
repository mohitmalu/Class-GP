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


def cls_bo(X, y, c, idx, thr, iterations, kernel, bounds, dimension, par, label_mat, partition_mat, w, beta_data,
           const, alpha_data, theta_initial, max_depth, initial_pt=None, n_restarts=1, delta=0.01, alpha=1,
           num_iter_refit=10, min_samp=2, restarts=1, normalize=True, weighted=True, std=0):
    if initial_pt is None:
        initial_pt = np.zeros((1, dimension))

    par2 = [par[j][1:-1] for j in range(dimension)]
    u = X.shape[0]  # Total number of partitions
    # u = np.array([len(X[clf_leaf_node_data[jj.leaf_id]]) for jj in clf_leaf_nodes])  # points in each partition
    beta = alpha * (2 * np.log((u ** 2) * 2 * (np.pi ** 2) / (3 * delta)) +
                    2 * dimension * np.log((u ** 2) * dimension * 20 * 1000 *
                                           np.sqrt(np.log(4 * dimension / delta))))
    # beta = 2.log(t^2.2.pi^2/(3.delta)) + 2.d.log(t^2.d.b.r(sqrt(log(4da/delta)))) r=20, b = 1000,
    # print("================= Fitting Tree and GP =============== ")
    clf, clf_leaf_nodes, clf_leaf_cls, clf_leaf_node_data, gps \
        = bo_hetero(X, y, c, idx, thr, kernel, bounds, theta_initial, max_depth, min_samp=min_samp,
                    restarts=restarts, normalize=normalize, weighted=weighted)
    n_iter = 0  # Iteration count compared with num_iter_refit after which retrain the tree and model
    for j in range(iterations):
        u = X.shape[0]  # Total number of partitions
        # u = np.array([len(X[clf_leaf_node_data[jj.leaf_id]]) for jj in clf_leaf_nodes])  # points in each partition
        beta = alpha * (2 * np.log((u ** 2) * 2 * (np.pi ** 2) / (3 * delta)) +
                        2 * dimension * np.log((u ** 2) * dimension * 20 * 1000 *
                                               np.sqrt(np.log(4 * dimension / delta))))
        # print("================= Finding next point of evaluation =============== ")
        x_proposed, x_ucb, x_leaf_index = propose_location(ucb_sampling, gps, clf_leaf_nodes, beta,
                                                           dimension, n_restarts, initial_pt)
        # print(x_proposed, -x_ucb, x_leaf_index)
        c_estimate = clf_leaf_cls[x_leaf_index]  # Cls estimate of new sample as per the generated tree
        c_eval = gen_cls_dataset(x_proposed, par, label_mat)  # True cls eval of the new sample
        p_eval = gen_cls_dataset(x_proposed, par, partition_mat)  # True partition eval of the new sample
        y_eval = np.array(
            gen_y_dataset_har(x_proposed, c_eval, p_eval, w, beta_data, const, alpha=alpha_data, std=std))  # func eval
        idx_eval, thr_eval = gen_idx_thr(x_proposed, par2)  # Index and Threshold of the new sample
        X = np.append(X, x_proposed, axis=0)
        c = np.append(c, c_eval)
        y = np.append(y, y_eval)
        idx = np.append(idx, idx_eval)
        thr = np.append(thr, thr_eval)
        # print (X.shape, c.shape, y.shape)
        if c_estimate == c_eval[0] and n_iter <= num_iter_refit:
            # print("================= Refitting just current GP =============== ")
            clf_leaf_node_data = leaf_node_data(X, clf.tree_)
            gps[clf_leaf_nodes[x_leaf_index]].fit(X[clf_leaf_node_data[clf_leaf_nodes[x_leaf_index].leaf_id]],
                                                  y[clf_leaf_node_data[clf_leaf_nodes[x_leaf_index].leaf_id]])
            cls_leaf_nodes = clf_leaf_nodes[clf_leaf_cls == c_eval[0]]
            gps_cls = [gps[jj] for jj in cls_leaf_nodes]
            gps_opt = [gps[jj] for jj in cls_leaf_nodes if hasattr(gps[jj], "kernel_")]
            gps_bounds = gps_cls[0].kernel.bounds
            if len(gps_opt) > 1:
                if not weighted:
                    def objective_func1(theta,
                                        eval_gradient=gradient_eval):  # Summing up individual objective functions
                        if eval_gradient:
                            lml = sum([objective_func(theta, gp, eval_gradient=eval_gradient)[0] for gp in gps_opt])
                            grad = sum([objective_func(theta, gp, eval_gradient=eval_gradient)[1] for gp in gps_opt])
                            return lml, grad
                        else:
                            return sum([objective_func(theta, gp, eval_gradient=eval_gradient) for gp in gps_opt])
                else:
                    def objective_func1(theta,
                                        eval_gradient=gradient_eval):  # Summing up individual objective functions
                        if eval_gradient:
                            lml = sum([gp.X_train_.shape[0] * objective_func(theta, gp, eval_gradient=eval_gradient)[0]
                                       for gp in gps_opt])
                            grad = sum([gp.X_train_.shape[0] * objective_func(theta, gp, eval_gradient=eval_gradient)[1]
                                        for gp in gps_opt])
                            return lml, grad
                        else:
                            return sum([gp.X_train_.shape[0] * objective_func(theta, gp, eval_gradient=eval_gradient)
                                        for gp in gps_opt])

                optimum = [(gps_opt[0]._constrained_optimization(objective_func1, theta_initial, bounds=gps_bounds))]
                if restarts > 0:  # Number of times the optimization is to be performed
                    if not np.isfinite(gps_opt[0].kernel.bounds).all():
                        raise ValueError("Multiple optimizer restarts (n_restarts_optimizer>0) "
                                         "requires that all bounds are finite.")
                    for iteration in range(restarts):
                        theta_initial1 = gps_opt[0]._rng.uniform(gps_bounds[:, 0], gps_bounds[:, 1])
                        optimum.append(
                            gps_opt[0]._constrained_optimization(objective_func1, theta_initial1, gps_bounds))

                # Select result from run with minimal (negative) log-marginal likelihood
                lml_values = list(map(itemgetter(1), optimum))
                for i in gps_opt:
                    i.kernel_.theta = optimum[np.argmin(lml_values)][0]
                    """Updating the gps with the new kernel hyper-parameters"""
                    K = i.kernel_(i.X_train_)
                    K[np.diag_indices_from(K)] += i.alpha
                    i.L_ = cholesky(K, lower=GPR_CHOLESKY_LOWER, check_finite=False)
                    # alpha_ = L^T \ (L \ y) which is used in prediction
                    i.alpha_ = cho_solve((i.L_, GPR_CHOLESKY_LOWER), i.y_train_, check_finite=False)
            n_iter += 1
        else:
            # print("================= Refitting Tree and GP =============== ")
            clf, clf_leaf_nodes, clf_leaf_cls, clf_leaf_node_data, gps \
                = bo_hetero(X, y, c, idx, thr, kernel, bounds, theta_initial, max_depth, min_samp=min_samp,
                            restarts=restarts, normalize=normalize, weighted=weighted)
            n_iter = 0
    return clf, gps, clf_leaf_nodes, clf_leaf_cls, beta, X, y, c


def gp_bo(X, y, iterations, kernel, bounds, dimension, par, label_mat, partition_mat, w, beta_data, const, alpha_data,
          theta_initial, initial_pt=None, n_restarts=1, delta=0.01, alpha=1, restarts=1, normalize=True, std=0):
    if initial_pt is None:
        initial_pt = np.zeros((1, dimension))
    m = X.shape[0]
    beta = alpha * (2 * np.log((m ** 2) * 2 * (np.pi ** 2) / (3 * delta)) +
                    2 * dimension * np.log((m ** 2) * dimension * 20 * 1000 *
                                           np.sqrt(np.log(4 * dimension / delta))))
    bo_gp = gaussian_process.GaussianProcessRegressor(clone(kernel), normalize_y=normalize,
                                                      n_restarts_optimizer=restarts)
    c = []
    for j in range(iterations):
        bo_gp.fit(X, y)
        m = X.shape[0]
        beta = alpha * (2 * np.log((m ** 2) * 2 * (np.pi ** 2) / (3 * delta)) +
                        2 * dimension * np.log((m ** 2) * dimension * 20 * 1000 *
                                               np.sqrt(np.log(4 * dimension / delta))))
        x_proposed, x_ucb = propose_location_gp(ucb_sampling, bo_gp, beta, dimension, n_restarts, initial_pt, bounds)
        # print(x_proposed, -x_ucb)

        c_eval = gen_cls_dataset(x_proposed, par, label_mat)
        p_eval = gen_cls_dataset(x_proposed, par, partition_mat)
        y_eval = np.array(gen_y_dataset_har(x_proposed, c_eval, p_eval, w, beta_data, const, alpha_data, std=std))

        X = np.append(X, x_proposed, axis=0)
        c += [c_eval]
        y = np.append(y, y_eval)
        # print (X.shape, y.shape)
    return bo_gp, beta, X, y, c


def cls_bo_multi(initial_budget, dimension, tup, bounds, k, n_partitions, w, beta_data, const, alpha_data, iterations,
                 kernel, theta_initial, max_depth, y_max, file_name, n_restarts=1, initial_pt=None, delta=0.01, alpha=1,
                 num_iter_refit=10, min_samp=2, restarts=1, normalize=True, weighted=True, std=0):
    cbo_rewards_run_iteration = []
    gpbo_rewards_run_iteration = []
    data = {}
    par, label_mat = checker_part_unbalanced(tup, dimension, k, bounds)
    partition_mat = np.arange(n_partitions).reshape(tup)
    par2 = [par[j][1:-1] for j in range(dimension)]
    print(par, '\n', label_mat, '\n', partition_mat, '\n')
    for j in range(runs):
        """Data Generation"""
        x_dataset = x_data(dimension, bounds, initial_budget)  # Generating Uniformly Random input data (X)
        c_dataset = gen_cls_dataset(x_dataset, par, label_mat)  # cls labels for the generated data
        p_dataset = gen_cls_dataset(x_dataset, par, partition_mat)  # partition labels for the generated data
        y_dataset = gen_y_dataset_har(x_dataset, c_dataset, p_dataset, w, beta_data, const,
                                      alpha_data, std=std)  # Output y_dataset
        idx, thr = gen_idx_thr(x_dataset, par2)
        # print(y_dataset.shape, x_dataset.shape)

        """Class BO"""
        print("=============== Running CBO ==================")
        t1 = time()
        clf, gps, clf_leaf_nodes, clf_leaf_cls, beta_cbo, x_cbo, y_cbo, c_cbo \
            = cls_bo(x_dataset, y_dataset, c_dataset, idx, thr, iterations, kernel, bounds, dimension, par,
                     label_mat, partition_mat, w, beta_data, const, alpha_data, theta_initial, max_depth,
                     n_restarts=n_restarts, initial_pt=initial_pt, delta=delta, alpha=alpha,
                     num_iter_refit=num_iter_refit,
                     min_samp=min_samp, restarts=restarts, normalize=normalize, weighted=weighted, std=std)
        t2 = time()
        cbo_rewards_run_iteration += [[max(y_cbo[initial_budget:initial_budget + jj + 1]) for jj in range(iterations)]]
        print("CBO Maximum", max(y_cbo[initial_budget:]), "and time taken = ", round(t2 - t1, 0))

        """Standard BO"""
        print("=============== Running Standard BO ==================")
        t3 = time()
        bo_gp, beta_gp, x_gp, y_gp, c_gp \
            = gp_bo(x_dataset, y_dataset, iterations, kernel, bounds, dimension, par, label_mat, partition_mat, w,
                    beta_data, const, alpha_data, theta_initial, n_restarts=n_restarts, initial_pt=None, delta=delta,
                    alpha=alpha, restarts=restarts, normalize=normalize, std=std)
        t4 = time()
        gpbo_rewards_run_iteration += \
            [[max(y_gp[initial_budget - 1:initial_budget + jj + 1]) for jj in range(iterations)]]
        print("GP_BO Maximum", max(y_gp[initial_budget:]), "and time taken = ", round(t4 - t3, 0))

        """Data"""
        round_data = {"X": x_dataset, "y": y_dataset, "clf_cbo": clf, "gps_cbo": gps, "bo_gp": bo_gp,
                      "x_cbo": x_cbo, "y_cbo": y_cbo, "x_gp": x_gp, "y_gp": y_gp, "beta_cbo": beta_cbo,
                      "beta_gp": beta_gp}
        data[j] = round_data
    df_cbo = pd.DataFrame(cbo_rewards_run_iteration)
    df_gpbo = pd.DataFrame(gpbo_rewards_run_iteration)
    excel_file_name1 = file_name + "_cbo.xlsx"
    excel_file_name2 = file_name + "_gpbo.xlsx"
    df_cbo.to_excel(excel_file_name1)
    df_gpbo.to_excel(excel_file_name2)
    cbo_desc = df_cbo.describe()
    gpbo_desc = df_gpbo.describe()
    plot_rewards(cbo_desc, gpbo_desc, iterations, y_max, 'CBO', 'BO', file_name)
    data_file_name = file_name + ".pkl"
    pickle_out = open(data_file_name, "wb")
    pickle.dump(data, pickle_out)
    pickle_out.close()
    return df_cbo, df_gpbo, data


if __name__ == '__main__':
    ''' Modeling for multiple runs '''

    # for k in [2,4,6,8]:
    #     for total_budget in [5050, 5500, 6000]:
    #         for tups in [(2,), (4,), (6,), (8,)]:
    #             rstate = 175  # Random State
    #             dt = datetime.now().strftime("%m%d%y%H%M%S")
    #             # total_budget = 6000  # number of data points exp - 1000, 2000, 5000
    #             # k = 8  # Number of classes - exp - 2, 4, 8
    #             dimension = 2  # dimension -  exp - 2, 4, 8
    #             bounds = [(-10, 10)]*dimension  # bounds on the input data - same as space_bounds
    #             tup = tups * dimension
    #             n_partitions = tup[0]**dimension  # Number of partitions - exp - 20, 40, 80
    #             max_depth = 20  # Max depth of the tree
    #             test_data_size = 5000 / total_budget
    #             runs = 50  # Number of Macro reps
    #             kernel_def = ConstantKernel(1) * RBF(np.ones((dimension,))) + WhiteKernel()  # kernel for learning
    #             theta_initial = np.zeros_like(kernel_def.theta)  # initialization of log hyperparameters
    #             n_restarts = 3  # No of restarts of minimize optimizer
    #             initial_pt = None  # Initial point for minimize optimizer
    #             restarts = 1  # No of restarts of gp - theta initial is uniform random
    #             alpha = 1  # Multiple for beta in UCB (exploration)
    #             num_iter_refit = 10  # Refit the tree after this number of iterations
    #             min_samp = 2  # minimum number of leaf samples
    #             normalize = True  # Normalization of y while training GP's
    #             weighted = False # Likelihood is weighted with samples per partition

    #             """Vectors for Data_gen Function"""
    #             prng = RandomState(rstate)
    #             het_ratio = 1  # factor Gap between 2 set of frequencies
    #             w1 = np.arange(1, het_ratio * k + 1, het_ratio).reshape(-1, 1) *\
    #                  prng.randn(1, dimension)  # Fixed Frequency vector
    #             beta = (prng.randn(k, dimension))  # Fixed vector
    #             alpha = 0.1  # exponential factor
    #             c = prng.randn(n_partitions, 1)  # Fixed Offset/Intercept

    #             """ File name and Main Function """
    #             file_name = "Experiments/noisy_levy_cen_mse_" + str(dimension) + "d_" + str(total_budget) + "N_" \
    #                         + str(k) + "c_" + str(n_partitions) + "p_rbf_test_" + str(int(test_data_size * 100)) \
    #                         + "_bounds" + str(bounds[0]) + "_" + str(bounds[1]) + "_" + str(runs) + "runs_" \
    #                         + str(het_ratio) + "het_" + str(alpha) + "alp_" + dt
    #             print(file_name)
    #             np.random.seed(rstate)
    #             try:
    #                 mse_df, data = bo_multi_test_che(total_budget, dimension, tup, k, n_partitions, bounds,
    #                                                  test_data_size, kernel_def, file_name, theta_initial, max_depth,
    #                                                  w1, beta, c, runs, alpha=alpha, min_samp=min_samp,
    #                                                  restarts=restarts, normalize=normalize, weighted=weighted)
    #             except Exception as e:
    #                 print (k, total_budget, tups)
    #                 print(file_name)
    #                 print(f"Unexpected {e=}, {type(e)=}")
    #                 continue
    #             df = mse_df.iloc[:, 0:3]
    #             plt_df = df.dropna()
    #             plot_err_bars(plt_df, k, n_partitions, het_ratio, file_name)
    #             plt.show()

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

    '''Multiple CBO runs'''

    # for k in [2,3,4,5,6,7,8]:
    #     for tups in [(2,), (3,), (4,), (5,), (6,), (7,), (8,)]:
    #         rstate = 175  # Random State
    #         dt = datetime.now().strftime("%m%d%y%H%M%S")
    #         initial_budget = 20   # number of data points exp - 1000, 2000, 5000
    #         dimension = 2  # dimension -  exp - 2, 4, 8
    #         bounds = [(-10, 10)]*dimension  # bounds on the input data - same as space_bounds
    #         tup = tups * dimension
    #         n_partitions = tup[0] ** dimension  # Number of partitions - exp - 20, 40, 80
    #         iterations = 180
    #         max_depth = 20  # Max depth of the tree
    #         runs = 10  # Number of Macro reps
    #         kernel_def = ConstantKernel(1) * RBF(np.ones((dimension,))) + WhiteKernel()  # kernel for learning
    #         theta_initial = np.zeros_like(kernel_def.theta)  # initialization of log hyperparameters
    #         y_max = 0

    #         """Vectors for Data_gen Function"""
    #         prng = RandomState(rstate)
    #         het_ratio = 1  # factor Gap between 2 set of frequencies
    #         w = np.arange(1, het_ratio * k + 1, het_ratio).reshape(-1, 1) *\
    #             prng.randn(1, dimension)  # Fixed Frequency vector
    #         alpha_data = 0.1  # exponential factor
    #         beta_data = (prng.randn(k, dimension))  # Fixed vector
    #         const = prng.randn(n_partitions, 1)  # Fixed Offset/Intercept

    #         """ File name and Main Function """
    #         file_name = "Experiments/noisy_cls_bo_" + str(dimension) + "d_" + str(initial_budget) + "N_" \
    #                     + str(k) + "c_" + str(n_partitions) + "p_rbf_iterations_" + str(iterations) \
    #                     + "_bounds" + str(bounds[0]) + "_" + str(bounds[1]) + "_" + str(runs) + "runs_" \
    #                     + str(het_ratio) + "het_" + str(alpha_data) + "alp_" + dt
    #         print(file_name)
    #         np.random.seed(rstate)
    #         try:
    #             df_cbo, df_gpbo = cls_bo_multi(initial_budget, dimension, tup, bounds, k, n_partitions, w,
    #                                         beta_data, const, alpha_data, iterations, kernel_def,
    #                                         theta_initial, max_depth, y_max, file_name)
    #         except:
    #             print (k, initial_budget, tups)
    #             print(file_name)
    #             continue

    ''' CBO run '''
    rstate = 175  # Random State
    dt = datetime.now().strftime("%m%d%y%H%M%S")
    initial_budget = 50  # number of data points exp - 1000, 2000, 5000
    dimension = 2  # dimension -  exp - 2, 4, 8
    bounds = [(-10, 10)] * dimension  # bounds on the input data - same as space_bounds
    k = 4  # Number of classes - exp - 2, 4, 8
    tup = (4,) * dimension
    n_partitions = tup[0] ** dimension  # Number of partitions - exp - 20, 40, 80
    iterations = 250  # No of iterations of cbo and bo
    max_depth = 20  # Max depth of the tree - used in bo_reg function for restriction on depth of regression tree
    runs = 5  # Number of Macro reps
    kernel_def = ConstantKernel(1) * RBF(np.ones((dimension,))) + WhiteKernel()  # kernel for learning
    n_restarts = 3  # No of restarts of minimize optimizer
    initial_pt = None  # Initial point for minimize optimizer
    restarts = 1  # No of restarts of gp - theta initial is uniform random
    theta_initial = np.zeros_like(kernel_def.theta)  # theta initial for minimization when number of restarts = 0
    delta = 0.01  # Probability of error
    alpha = 1  # Multiple for beta in UCB (exploration)
    num_iter_refit = 10  # Refit the tree after this number of iterations
    min_samp = 2  # minimum number of leaf samples
    normalize = True  # Normalization of y while training GP's
    weighted = False  # Likelihood is weighted with samples per partition

    """Parameters for Data_gen Function"""
    prng = RandomState(rstate)
    het_ratio = 1  # factor Gap between 2 set of frequencies
    w = np.arange(1, het_ratio * k + 1, het_ratio).reshape(-1, 1) * prng.randn(1, dimension)  # Fixed Frequency vector
    alpha_data = 0.1  # exponential factor
    beta_data = (prng.randn(k, dimension))  # Fixed vector
    const = prng.permutation(n_partitions).reshape(-1, 1)  # Fixed Offset/Intercept decreasing
    const[0] = n_partitions + 4
    std = 0
    y_max = max(const) + 1  # Maximum function evaluation
    # const = prng.randn(n_partitions, 1)  # Fixed Offset/Intercept

    """ File name and Main Function """
    file_name = "Experiments/noiseless_non_weighted_har_cls_bo_" + str(dimension) + "d_" + str(initial_budget) + "N_" \
                + str(k) + "c_" + str(n_partitions) + "p_rbf_iterations_" + str(iterations) \
                + "_bounds" + str(bounds[0]) + "_" + str(bounds[1]) + "_" + str(runs) + "runs_" \
                + str(het_ratio) + "het_" + str(alpha_data) + "alp_" + str(std) + "std" + dt
    print(file_name)
    np.random.seed(rstate)
    try:
        df_cbo, df_gpbo, data = cls_bo_multi(initial_budget, dimension, tup, bounds, k, n_partitions, w, beta_data,
                                             const, alpha_data, iterations, kernel_def, theta_initial, max_depth, y_max,
                                             file_name, n_restarts=n_restarts, initial_pt=initial_pt, delta=delta,
                                             alpha=alpha, num_iter_refit=num_iter_refit, min_samp=min_samp,
                                             restarts=restarts, normalize=normalize, weighted=weighted, std=std)
    except Exception as e:
        print(k, initial_budget, tup)
        print(f"Unexpected {e}, {type(e)}")
