import numpy as np
from qbase import AC, CC, HDy, EDy, HDX, EDX

import os, glob
import pandas as pd

pd.set_option('display.float_format', lambda x: '%.5f' % x)

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier

from utils import create_bags_with_multiple_prevalence, binary_kl_divergence, absolute_error
from utils import g_mean, normalize
from sklearn.metrics import zero_one_loss, brier_score_loss    #NUEVO

import warnings
from sklearn.exceptions import DataConversionWarning
from pandas.core.common import SettingWithCopyWarning

warnings.simplefilter("ignore", DataConversionWarning)
warnings.simplefilter("ignore", SettingWithCopyWarning)


def indices_to_one_hot(data, n_classes):   #NUEVO
    """Convert an iterable of indices to one-hot encoded labels."""
    targets = np.array(data).reshape(-1)
    return np.eye(n_classes)[targets]


def main():

    # configuration params
    num_reps = 40
    num_bags = 50
    num_folds= 50
    master_seed = 2032

    estimator_grid = {
        "n_estimators": [10, 20, 40, 70, 100, 200, 250, 500],
        "max_depth": [1, 5, 10, 15, 20, 25, 30],
        "min_samples_leaf": [1, 2, 5, 10, 20]}


    datasets_dir = "./datasets"
    dataset_files = [file for file in glob.glob(os.path.join(datasets_dir, "*.csv"))]

    #dataset_files = ["./datasets/iris.3.csv"]

    dataset_names = [os.path.split(name)[-1][:-4] for name in dataset_files]
    print("There are a total of {} datasets.".format(len(dataset_names)))

    filename_out = "./results/results_UCI_" + str(num_reps) + "x" + str(num_bags)

    methods = ['AC', 'CC', 'EDX', 'EDy', 'HDX', 'HDy']
    total_errors_df = []
    for rep in range(num_reps):
        for dname, dfile in zip(dataset_names, dataset_files):
            current_seed = master_seed + rep
            print("*** Training over {}, rep {}".format(dname, rep + 1))
            total_errors_df.append(train_on_a_dataset(methods, dname, dfile, filename_out, estimator_grid,
                                                      master_seed, current_seed, num_bags, num_folds))

    total_errors_df = pd.concat(total_errors_df)
    total_errors_df.to_csv(filename_out+  "_all.csv", index=None)

    means_df = total_errors_df.groupby(['dataset', 'method'])[['mae']].agg(["mean"]).unstack().round(5)
    means_df.to_csv(filename_out+ "_means_mae.csv", header=methods)
    means_df2 = total_errors_df.groupby(['dataset', 'method'])[['error_clf']].agg(["mean"]).unstack().round(5)
    means_df2.to_csv(filename_out+ "_means_error.csv", header=methods)
    means_df3 = total_errors_df.groupby(['dataset', 'method'])[['brier_clf']].agg(["mean"]).unstack().round(5)
    means_df3.to_csv(filename_out+ "_means_brier.csv", header=methods)

    print(means_df)


def load_data(dfile, current_seed):

    df = pd.read_csv(dfile, header=None)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values.astype(np.int)
    if -1 in np.unique(y):
        y[y == -1] = 0

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=current_seed)
    X_train, X_test = normalize(X_train, X_test)
    return X_train, X_test, y_train, y_test



def select_estimator(X_train, y_train, estimator_grid, master_seed, current_seed):

    clf_ = RandomForestClassifier(random_state=master_seed, class_weight='balanced')
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=current_seed)
    gs = GridSearchCV(clf_, param_grid=estimator_grid, verbose=False, cv=skf, scoring=g_mean, n_jobs=-1, iid=False)
    gs.fit(X_train, y_train)
    #print("Best grid params:", gs.best_params_)
    clf = gs.best_estimator_
    return clf


def train_on_a_dataset(methods, dname, dfile, filename_out,  estimator_grid,
                       master_seed, current_seed, num_bags, num_folds):

    columns = ['dataset', 'method', 'truth', 'predictions', 'mae', "error_clf", "brier_clf" ]
    errors_df = pd.DataFrame(columns=columns)

    X_train, X_test, y_train, y_test = load_data(dfile, current_seed)
    folds = np.min([num_folds, np.min(np.unique(y_train, return_counts=True)[1])])

    clf = select_estimator(X_train, y_train, estimator_grid, master_seed, current_seed)

    ac = AC(estimator=clf)
    ac.fit(X_train, y_train, cv=folds)

    cc = CC(estimator=clf, sys_trained=ac)
    cc.fit(X_train, y_train, cv=folds)

    edy = EDy(estimator=clf, sys_trained=ac)
    edy.fit(X_train, y_train, cv=folds)

    hdy = HDy(b=8, bin_strategy='tasche', estimator=clf, sys_trained=ac)
    hdy.fit(X_train, y_train, cv=folds)

    edx = EDX()
    edx.fit(X_train, y_train)

    hdx = HDX(b=8, bin_strategy='tasche')
    hdx.fit(X_train, y_train)

    for n_bag, (X_test_, y_test_, prev_true) in enumerate(
               create_bags_with_multiple_prevalence(X_test, y_test, num_bags, current_seed)):

        pred_test_ = clf.predict_proba(X_test_)  #prediccion de clasificación  #NUEVO
        # Error
        error_clf = zero_one_loss(np.array(y_test_), np.argmax(pred_test_,axis=1))  # --> GUARDAR
        # Brier loss
        brier_clf = brier_score_loss(indices_to_one_hot(y_test_, 2)[:, 0], pred_test_[:, 0])  # --> GUARDAR

        prev_true = prev_true[1]
        prev_preds = [
            ac.predict(X_test_)[1],
            cc.predict(X_test_)[1],
            edx.predict(X_test_)[1],
            edy.predict(X_test_)[1],
            hdx.predict(X_test_)[1],
            hdy.predict(X_test_)[1]
        ]
        for n_method, (method, prev_pred) in enumerate(zip(methods, prev_preds)):
            mae = absolute_error(prev_true, prev_pred)
            errors_df = errors_df.append(
                pd.DataFrame([[dname, method, prev_true, prev_pred, mae, error_clf, brier_clf]], columns=columns))

    # uncomment if you want to save intermediate results
    # errors_df.to_csv(filename_out + "_bak.csv", mode='a', index=None)

    return errors_df


if __name__ == '__main__':
    main()


