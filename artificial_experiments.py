import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import zero_one_loss, brier_score_loss
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from qbase import CC, AC, EDX, EDy, HDX, HDy
from utils import absolute_error


def indices_to_one_hot(data, n_classes):
    """Convert an iterable of indices to one-hot encoded labels."""
    targets = np.array(data).reshape(-1)
    return np.eye(n_classes)[targets]


def run_experiment(est_name, seed, dim, param, ntrain, ntest, nreps, nbags, nfolds, save_all):
    """ Run a single experiment

        Parameters
        ----------
        est_name: str
            Name of the estimator. 'LR' or 'SVM-RBF'

        seed: int
            Seed of the experiment

        dim: int
            Dimension of the dataset, 1 or 2

        param: int, str
            Extra oarameter for the definition of the problem.
            If dim==1, this value is the std.
            If dim=2 this value is an string to indicate if the dataset is the one designed to test HDX

        ntrain : list
            List with the number of training examples that must be tested, e.g.,[50, 100, 200]

        ntest: int
            Number of testing instances in each bag

        nreps: int
            Number of training datasets created

        nbags: int
            Number of testing bags created for each training datasets.
            The total number of experiments will be nreps * nbags

        nfolds: int
            Number of folds used to estimate the training distributions by the methods AC, HDy and EDy

        save_all: bool
            True if the results of each single experiment must be saved
    """

    # range of testing prevalences
    low = round(ntest * 0.05)
    high = round(ntest * 0.95)

    if est_name == 'LR':
        estimator = LogisticRegression(C=1, random_state=seed, max_iter=10000, solver='liblinear')
    else:
        estimator = SVC(C=1, kernel='rbf', random_state=seed, max_iter=10000, gamma=0.2, probability=True)

    rng = np.random.RandomState(seed)

    #   methods
    methods_names = ['AC', 'CC', 'EDX', 'EDy', 'HDX', 'HDy']
    #   to store all the quant_results
    quant_results = np.zeros((len(methods_names), len(ntrain)))
    classif_results = np.zeros((2, len(ntrain)))

    std1 = std2 = mu3 = mu4 = cov1 = cov2 = cov3 = cov4 = 0
    if dim == 1:
        # 1D
        mu1 = -1
        std1 = param
        mu2 = 1
        std2 = std1
    else:
        # 2D
        mu1 = [-1.00, 1.00]
        mu2 = [1.00, 1.00]
        mu3 = [1.00, -1.00]

        cov1 = [[0.4, 0],
                [0, 0.4]]
        cov2 = cov1
        cov3 = cov1

        x1 = rng.multivariate_normal(mu1, cov1, 400)
        x3 = rng.multivariate_normal(mu3, cov3, 400)

        plt.scatter(np.vstack((x1[:, 0], x3[:, 0])), np.vstack((x1[:, 1], x3[:, 1])), c='r', marker='+', s=12,
                    label='Class \u2212' + '1')

        if param == 'HDX':
            mu4 = [-1.00, -1.00]
            cov4 = cov1

            x2 = rng.multivariate_normal(mu2, cov2, 400)
            x4 = rng.multivariate_normal(mu4, cov4, 400)

            plt.scatter(np.vstack((x2[:, 0], x4[:, 0])), np.vstack((x2[:, 1], x4[:, 1])),
                        c='b', marker='x', s=8, label='Class +1')
        else:
            x2 = rng.multivariate_normal(mu2, cov2, 800)

            plt.scatter(x2[:, 0], x2[:, 1], c='b', marker='x', s=8, label='Class +1')

        plt.xlabel('$x_1$')
        plt.ylabel('$x_2$')
        plt.legend(loc='best')
        plt.savefig('./results/artificial-2D-' + param + '.png', dpi=300)

    for k in range(len(ntrain)):

        all_quant_results = np.zeros((len(methods_names), nreps * nbags))
        all_classif_results = np.zeros((2, nreps * nbags))

        print()
        print('#Training examples ', ntrain[k], 'Rep#', end=' ')

        for rep in range(nreps):

            print(rep+1, end=' ')

            if dim == 1:
                x_train = np.vstack(((std1 * rng.randn(ntrain[k], 1) + mu1), (std2 * rng.randn(ntrain[k], 1) + mu2)))
            else:
                if param == 'HDX':
                    x_train = np.vstack((rng.multivariate_normal(mu1, cov1, ntrain[k] // 2),
                                         rng.multivariate_normal(mu3, cov3, ntrain[k] - ntrain[k] // 2),
                                         rng.multivariate_normal(mu2, cov2, ntrain[k] // 2),
                                         rng.multivariate_normal(mu4, cov4, ntrain[k] - ntrain[k] // 2)))
                else:
                    x_train = np.vstack((rng.multivariate_normal(mu1, cov1, ntrain[k] // 2),
                                         rng.multivariate_normal(mu3, cov3, ntrain[k] - ntrain[k] // 2),
                                         rng.multivariate_normal(mu2, cov2, ntrain[k])))

            y_train = np.hstack((np.zeros(ntrain[k], dtype=int), np.ones(ntrain[k], dtype=int)))

            #  estimator for estimating the training distribution, CV 50
            ac = AC(estimator=estimator)
            ac.fit(x_train, y_train, cv=nfolds)

            cc = CC(estimator=estimator, sys_trained=ac)
            cc.fit(x_train, y_train, cv=nfolds)

            edx = EDX()
            edx.fit(x_train, y_train)

            edy = EDy(estimator=estimator, sys_trained=ac)
            edy.fit(x_train, y_train, cv=nfolds)

            hdx = HDX(b=8, bin_strategy='tasche')
            hdx.fit(x_train, y_train)

            hdy = HDy(b=8, bin_strategy='tasche', estimator=estimator, sys_trained=ac)
            hdy.fit(x_train, y_train, cv=nfolds)

            for n_bag in range(nbags):

                ps = rng.randint(low, high, 1)
                ps = np.append(ps, [0, ntest])
                ps = np.diff(np.sort(ps))

                if dim == 1:
                    x_test = np.vstack(((std1 * rng.randn(ps[0], 1) + mu1), (std2 * rng.randn(ps[1], 1) + mu2)))
                else:
                    if param == 'HDX':
                        x_test = np.vstack((rng.multivariate_normal(mu1, cov1, ps[0] // 2),
                                            rng.multivariate_normal(mu3, cov3, ps[0] - ps[0] // 2),
                                            rng.multivariate_normal(mu2, cov2, ps[1] // 2),
                                            rng.multivariate_normal(mu4, cov4, ps[1] - ps[1] // 2)))
                    else:
                        x_test = np.vstack((rng.multivariate_normal(mu1, cov1, ps[0] // 2),
                                            rng.multivariate_normal(mu3, cov3, ps[0] - ps[0] // 2),
                                            rng.multivariate_normal(mu2, cov2, ps[1])))

                y_test = np.hstack((np.zeros(ps[0], dtype=int), np.ones(ps[1], dtype=int)))

                pred_test = estimator.predict_proba(x_test)
                # Error
                all_classif_results[0, rep * nbags + n_bag] = zero_one_loss(np.array(y_test),
                                                                            np.argmax(pred_test, axis=1))
                # Brier loss
                all_classif_results[1, rep * nbags + n_bag] = brier_score_loss(indices_to_one_hot(y_test, 2)[:, 0],
                                                                               pred_test[:, 0])

                classif_results[0, k] = classif_results[0, k] + all_classif_results[0, rep * nbags + n_bag]

                classif_results[1, k] = classif_results[1, k] + all_classif_results[1, rep * nbags + n_bag]

                prev_true = ps[1] / ntest

                all_quant_results[0, rep * nbags + n_bag] = absolute_error(prev_true, ac.predict(x_test)[1])
                all_quant_results[1, rep * nbags + n_bag] = absolute_error(prev_true, cc.predict(x_test)[1])
                all_quant_results[2, rep * nbags + n_bag] = absolute_error(prev_true, edx.predict(x_test)[1])
                all_quant_results[3, rep * nbags + n_bag] = absolute_error(prev_true, edy.predict(x_test)[1])
                all_quant_results[4, rep * nbags + n_bag] = absolute_error(prev_true, hdx.predict(x_test)[1])
                all_quant_results[5, rep * nbags + n_bag] = absolute_error(prev_true, hdy.predict(x_test)[1])

                for nmethod, method in enumerate(methods_names):
                    quant_results[nmethod, k] = quant_results[nmethod, k] + \
                                                all_quant_results[nmethod, rep * nbags + n_bag]

        if save_all:
            name_file = './results/artificial-all' + str(dim) + 'D-' + str(param) + '-' + est_name + \
                        '-rep' + str(nreps) + '-value' + str(ntrain[k]) + '-ntest' + str(ntest) + '.txt'
            file_all = open(name_file, 'w')

            for method_name in methods_names:
                file_all.write('%s,' % method_name)
            file_all.write('Error, Brier loss')
            file_all.write('\n')
            for nrep in range(nreps):
                for n_bag in range(nbags):
                    for n_method in range(len(methods_names)):
                        file_all.write('%.5f, ' % all_quant_results[n_method, nrep * nbags + n_bag])
                    file_all.write('%.5f, ' % all_classif_results[0, nrep * nbags + n_bag])
                    file_all.write('%.5f, ' % all_classif_results[1, nrep * nbags + n_bag])
                    file_all.write('\n')
            file_all.close()

    quant_results = quant_results / (nreps * nbags)
    classif_results = classif_results / (nreps * nbags)

    name_file = './results/artificial-avg' + str(dim) + 'D-' + str(param) + '-' + est_name + '-rep' + str(nreps) + \
                '-ntest' + str(ntest) + '.txt'
    file_avg = open(name_file, 'w')
    file_avg.write('#examples, Error, ')
    for index, m in enumerate(methods_names):
        file_avg.write('%s, ' % m)
    file_avg.write('BrierLoss')
    for index, number in enumerate(ntrain):
        file_avg.write('\n%d, ' % number)
        # Error
        file_avg.write('%.5f, ' % classif_results[0, index])
        for i in quant_results[:, index]:
            file_avg.write('%.5f, ' % i)
        # Brier loss
        file_avg.write('%.5f' % classif_results[1, index])

    file_avg.close()


# MAIN
# 1D
run_experiment(est_name='LR', seed=42, dim=1, param=0.5, ntrain=[50, 100, 200, 500, 1000, 2000], ntest=200,
               nreps=40, nbags=50, nfolds=50, save_all=False)
run_experiment(est_name='LR', seed=42, dim=1, param=0.75, ntrain=[50, 100, 200, 500, 1000, 2000], ntest=200,
               nreps=40, nbags=50, nfolds=50, save_all=False)
run_experiment(est_name='LR', seed=42, dim=1, param=1.0, ntrain=[50, 100, 200, 500, 1000, 2000], ntest=200,
               nreps=40, nbags=50, nfolds=50, save_all=False)

run_experiment(est_name='LR', seed=42, dim=1, param=0.5, ntrain=[50, 100, 200, 500, 1000, 2000], ntest=2000,
               nreps=40, nbags=50, nfolds=50, save_all=False)
run_experiment(est_name='LR', seed=42, dim=1, param=0.75, ntrain=[50, 100, 200, 500, 1000, 2000], ntest=2000,
               nreps=40, nbags=50, nfolds=50, save_all=False)
run_experiment(est_name='LR', seed=42, dim=1, param=1.0, ntrain=[50, 100, 200, 500, 1000, 2000], ntest=2000,
               nreps=40, nbags=50, nfolds=50, save_all=False)

# 2D
run_experiment(est_name='LR', seed=42, dim=2, param='', ntrain=[50, 100, 200, 500, 1000, 2000], ntest=2000,
              nreps=40, nbags=50, nfolds=50, save_all=False)
run_experiment(est_name='SVM-RBF', seed=42, dim=2, param='', ntrain=[50, 100, 200, 500, 1000, 2000], ntest=2000,
              nreps=40, nbags=50, nfolds=50, save_all=False)

run_experiment(est_name='SVM-RBF', seed=42, dim=2, param='HDX', ntrain=[50, 100, 200, 500, 1000, 2000], ntest=2000,
              nreps=40, nbags=50, nfolds=50, save_all=False)
