from abc import ABCMeta, abstractmethod
import six
import warnings
import math
import numpy as np
from scipy.stats import norm
import quadprog
from scipy.stats import rankdata
from sklearn.base import BaseEstimator
from sklearn.metrics import confusion_matrix, euclidean_distances
from sklearn.model_selection import GridSearchCV, cross_val_predict, StratifiedKFold
from sklearn.utils import check_X_y
from utils import solve_hd, is_pd, nearest_pd


class BaseQ (six.with_metaclass(ABCMeta, BaseEstimator)):
    def __init__(self, estimator):
        self.estimator_ = estimator
        self.predictions_train_ = []
        self.classes_ = []

    @abstractmethod
    def predict(self, X):
        """Predict the prevalence"""

    def is_fitted(self):
        return self.classes_ != []


    def fit(self, X, y, cv=5):
        if self.is_fitted():
            return

        X, y = check_X_y(X, y, accept_sparse=True)

        self.classes_ = np.unique(y).tolist()

        self.estimator_ = self.estimator_.fit(X, y)
        if isinstance(self.estimator_, GridSearchCV):
                self.estimator_= self.estimator_.best_estimator_

        if (cv!=None and cv>=2):
            skf = StratifiedKFold(n_splits=cv,  shuffle=True, random_state=2032)
            results = cross_val_predict(self.estimator_, X, y, cv=skf, method="predict_proba")
            self.predictions_train_ = results[:,1]  #probabilities of class 1
        else:
            raise ValueError("Invalid value for cv param")
        return self

class CC(BaseQ):  #Nuevo 1/11/2019
    def __init__(self, estimator , sys_trained = None):
        super(CC, self).__init__(estimator)

        if sys_trained!=None:
            if isinstance(sys_trained, BaseQ):
                self.estimator_ = sys_trained.estimator_
                self.predictions_train_ = sys_trained.predictions_train_
                self.classes_ = sys_trained.classes_
            else:
                raise  TypeError("Invalid type for sys_trained param")


    def fit(self, X, y, cv=5):
        super().fit(X, y, cv)

    def predict(self, X):
        predictions = self.estimator_.predict(X)
        freq = np.bincount(predictions, minlength=2)
        relative_freq = freq / float(np.sum(freq))

        probabilities = np.clip(relative_freq[1], 0, 1)
        probabilities = np.array([1 - probabilities, probabilities])
        if np.sum(probabilities) == 0:
            return probabilities
        return probabilities / np.sum(probabilities)



class AC(BaseQ):

    def __init__(self, estimator , sys_trained = None):
        super(AC, self).__init__(estimator)

        if sys_trained!=None:
            if isinstance(sys_trained, BaseQ):
                self.estimator_ = sys_trained.estimator_
                self.predictions_train_ = sys_trained.predictions_train_
                self.classes_ = sys_trained.classes_
            else:
                raise  TypeError("Invalid type for sys_trained param")



    def fit(self, X, y, cv=5):
        super().fit(X, y, cv)

        preds = np.copy(self.predictions_train_)
        preds[np.where(preds < 0.5)] = 0
        preds[np.where(preds >= 0.5)] = 1
        tn, fp, fn, tp = confusion_matrix(y, preds).ravel()
        self.tpr_ = tp / (tp + fn)
        self.fpr_ = fp / (fp + tn)


    def predict(self, X):
        predictions = self.estimator_.predict(X)
        freq = np.bincount(predictions, minlength=2)
        relative_freq = freq / float(np.sum(freq))
        adjusted = (relative_freq - self.fpr_) / float(self.tpr_ - self.fpr_)
        adjusted = np.nan_to_num(adjusted)
        probabilities = np.clip(adjusted[1], 0, 1)
        probabilities = np.array([1 - probabilities, probabilities])
        if np.sum(probabilities) == 0:
            return probabilities
        return probabilities / np.sum(probabilities)


class HDy(BaseQ):

    def __init__(self, estimator, sys_trained=None, b=8, bin_strategy='equal'):
        super(HDy, self).__init__(estimator)

        if sys_trained!=None:
            if isinstance(sys_trained, BaseQ):
                self.estimator_ = sys_trained.estimator_
                self.predictions_train_ = sys_trained.predictions_train_
                self.classes_ = sys_trained.classes_
            else:
                raise  TypeError("Invalid type for sys_trained param")

        self.b = b
        self.bin_strategy = bin_strategy


    def fit(self, X, y, cv=5):
        if not self.b:
            return

        super().fit(X, y, cv)

        n_classes = len(self.classes_)

        preds = self.predictions_train_
        pos_preds = preds[y == 1]
        neg_preds = preds[y == 0]

        self.train_dist_ = np.zeros((self.b, n_classes))

        if self.bin_strategy == 'equal':
            self.bincuts = np.histogram_bin_edges(preds, bins=self.b, range=(0., 1.))
        elif self.bin_strategy == 'tasche':
            # only for binary quantification
            mu = (np.mean(neg_preds) + np.mean(pos_preds)) / 2
            std = (np.std(neg_preds) + np.std(pos_preds)) / 2
            if std > 0:
                self.bincuts = [std * norm.ppf(i/self.b) + mu for i in range(0, self.b+1)]
            else:
                self.bincuts = np.histogram_bin_edges(preds, bins=self.b, range=[0., 1.])

        self.train_dist_[:, 0], _ = np.histogram(neg_preds, bins=self.bincuts)
        self.train_dist_[:, 1], _ = np.histogram(pos_preds, bins=self.bincuts)
        self.train_dist_[:, 0] = self.train_dist_[:, 0] / float(np.sum(y == 0))
        self.train_dist_[:, 1] = self.train_dist_[:, 1] / float(np.sum(y == 1))




    def predict(self, X):
        if not self.b:
            raise ValueError("If HDy predictions are in order, the quantifier must be trained with the parameter `b`")

        n_classes = len(self.classes_)
        preds = self.estimator_.predict_proba(X)[:, 1]
        
        # NEW
        test_dist = np.zeros((self.b, 1))
        test_dist[:, 0], _ = np.histogram(preds, bins=self.bincuts)
        test_dist = test_dist / float(X.shape[0])
        
        # OLD
        # pdf, _ = np.histogram(preds, self.b, range=(0, 1))
        # test_dist = pdf / float(X.shape[0])
        # test_dist = np.expand_dims(test_dist, -1)
        
        return solve_hd(self.train_dist_, test_dist, n_classes)


class EDy(BaseQ):

    def __init__(self, estimator, sys_trained = None):
        super(EDy, self).__init__(estimator)

        if sys_trained!=None:
            if isinstance(sys_trained, BaseQ):
                self.estimator_ = sys_trained.estimator_
                self.predictions_train_ = sys_trained.predictions_train_
                self.classes_ = sys_trained.classes_
            else:
                raise  TypeError("Invalid type for sys_trained param")

    def fit(self, X, y, cv=5):
        super().fit(X, y, cv)

        self.X_train = X
        self.y_train = y
        self.train_dist_ = dict.fromkeys(self.classes_)
        preds = self.predictions_train_
        pos_preds = preds[y == 1]
        neg_preds = preds[y == 0]
        self.train_dist_[0] = neg_preds
        self.train_dist_[1] = pos_preds


    def predict(self, X, use_models_est_dist=False):
        n_classes = len(self.classes_)

        K = np.zeros((n_classes, n_classes))
        Kt = np.zeros(n_classes)

        repr_train = self.train_dist_
        repr_test = self.estimator_.predict_proba(X)[:, 1]

        for i in range(n_classes):
            K[i, i] = self.l1_norm(repr_train[self.classes_[i]],
                                   repr_train[self.classes_[i]])
            Kt[i] = self.l1_norm(repr_train[self.classes_[i]], repr_test)
            for j in range(i + 1, n_classes):
                K[i, j] = self.l1_norm(repr_train[self.classes_[i]],
                                       repr_train[self.classes_[j]])
                K[j, i] = K[i, j]

        train_cls = np.array(np.bincount(self.y_train))[:, np.newaxis]
        train_cls_m = np.dot(train_cls, train_cls.T)
        m = float(len(X))

        K = K / train_cls_m
        Kt = Kt / (train_cls.squeeze() * m)
        B = np.zeros((n_classes - 1, n_classes - 1))
        for i in range(n_classes - 1):
            B[i, i] = - K[i, i] - K[-1, -1] + 2 * K[i, -1]
            for j in range(n_classes - 1):
                if j == i:
                    continue
                B[i, j] = - K[i, j] - K[-1, -1] + K[i, -1] + K[j, -1]

        t = - Kt[:-1] + K[:-1, -1] + Kt[-1] - K[-1, -1]

        G = 2 * B
        if not is_pd(G):
            G = nearest_pd(G)

        a = 2 * t
        C = -np.vstack([np.ones((1, n_classes - 1)), -np.eye(n_classes - 1)]).T
        b = -np.array([1] + [0] * (n_classes - 1), dtype=np.float)
        sol = quadprog.solve_qp(G=G, a=a, C=C, b=b)

        p = sol[0]
        p = np.append(p, 1 - p.sum())
        return p


    def l1_norm(self, p, q):
        return np.abs(p[:, None] - q).sum()


class CvMy(BaseQ):

    def __init__(self, estimator, sys_trained = None):
        super(CvMy, self).__init__(estimator)

        if sys_trained!=None:
            if isinstance(sys_trained, BaseQ):
                self.estimator_ = sys_trained.estimator_
                self.predictions_train_ = sys_trained.predictions_train_
                self.classes_ = sys_trained.classes_
            else:
                raise  TypeError("Invalid type for sys_trained param")


    def fit(self, X, y, cv=5):
        super().fit(X, y, cv)

        self.X_train = X
        self.y_train = y
        self.train_repr = self.predictions_train_[:, np.newaxis]


    def predict(self, X, use_models_est_dist=False):
        n_classes = len(self.classes_)

        test_repr = self.estimator_.predict_proba(X)[..., 1][:, np.newaxis]

        Hn = rankdata(np.concatenate(np.concatenate([self.train_repr, test_repr])))
        Htr = Hn[:len(self.X_train)]
        Htst = Hn[len(self.X_train):]

        K = np.zeros((n_classes, n_classes))
        Kt = np.zeros(n_classes)

        for i in range(n_classes):
            K[i, i] = self.distance(Htr[self.y_train == self.classes_[i]],
                                    Htr[self.y_train == self.classes_[i]])

            Kt[i] = self.distance(Htr[self.y_train == self.classes_[i]], Htst)
            for j in range(i + 1, n_classes):
                K[i, j] = self.distance(Htr[self.y_train == self.classes_[i]],
                                        Htr[self.y_train == self.classes_[j]])
                K[j, i] = K[i, j]

        train_cls = np.array(np.bincount(self.y_train))[:, np.newaxis]
        train_cls_m = np.dot(train_cls, train_cls.T)
        m = float(len(X))

        K = K / train_cls_m
        Kt = Kt / (train_cls.squeeze() * m)
        B = np.zeros((n_classes - 1, n_classes - 1))
        for i in range(n_classes - 1):
            B[i, i] = - K[i, i] - K[-1, -1] + 2 * K[i, -1]
            for j in range(n_classes - 1):
                if j == i:
                    continue
                B[i, j] = - K[i, j] - K[-1, -1] + K[i, -1] + K[j, -1]

        t = - Kt[:-1] + K[:-1, -1] + Kt[-1] - K[-1, -1]

        G = 2 * B
        if not is_pd(G):
            G = nearest_pd(G)

        a = 2 * t
        C = np.vstack([- np.ones((1, n_classes - 1)), np.eye(n_classes - 1)]).T
        b = np.array([-1] + [0] * (n_classes - 1), dtype=np.float)
        sol = quadprog.solve_qp(G=G, a=a, C=C, b=b)

        p = sol[0]
        p = np.append(p, 1 - p.sum())

        return p


    def distance(self, p, q):
        return np.abs(p[:, None] - q).sum()



class EDX(six.with_metaclass(ABCMeta, BaseEstimator)):
    def __init__(self):
        pass

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        self.K = np.zeros((n_classes, n_classes))
        self.X_train = X
        self.y_train = y

        for i in range(n_classes):
            self.K[i, i] = self.distance(X[y == self.classes_[i]],
                                         X[y == self.classes_[i]])
            for j in range(i + 1, n_classes):
                self.K[i, j] = self.distance(X[y == self.classes_[i]],
                                             X[y == self.classes_[j]])
                self.K[j, i] = self.K[i, j]

    def predict(self, X, method="edx"):
        assert method == "edx"
        n_classes = len(self.classes_)

        Kt = np.zeros(n_classes)

        for i in range(n_classes):
            Kt[i] = self.distance(self.X_train[self.y_train == self.classes_[i]], X)

        train_cls = np.array(np.bincount(self.y_train))[:, np.newaxis]
        train_cls_m = np.dot(train_cls, train_cls.T)
        m = float(len(X))

        K = self.K / train_cls_m
        Kt = Kt / (train_cls.squeeze() * m)
        B = np.zeros((n_classes - 1, n_classes - 1))
        for i in range(n_classes - 1):
            B[i, i] = - K[i, i] - K[-1, -1] + 2 * K[i, -1]
            for j in range(n_classes - 1):
                if j == i:
                    continue
                B[i, j] = - K[i, j] - K[-1, -1] + K[i, -1] + K[j, -1]

        t = - Kt[:-1] + K[:-1, -1] + Kt[-1] - K[-1, -1]

        G = B
        if not is_pd(G):
            G = nearest_pd(G)

        a = t
        C = -np.vstack([np.ones((1, n_classes - 1)), -np.eye(n_classes - 1)]).T
        b = -np.array([1] + [0] * (n_classes - 1), dtype=np.float)
        sol = quadprog.solve_qp(G=G, a=a, C=C, b=b)

        p = sol[0]
        p = np.append(p, 1 - p.sum())

        return p

    def distance(self, p, q):
        return euclidean_distances(p, q).sum()


class HDX(six.with_metaclass(ABCMeta, BaseEstimator)):

    def __init__(self, b=8, bin_strategy='equal'):
        self.b = b
        self.bin_strategy = bin_strategy

    def fit(self, X, y):
        # warnings.simplefilter('error')
        # try:
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        # NEW
        if self.bin_strategy == 'equal':
            att_ranges = [(a.min(), a.max()) for a in X.T]
            self.bincuts = np.zeros((X.shape[1], self.b + 1))
            # self.bincuts[:, 0] = -np.inf
            # self.bincuts[:, -1] = np.inf
            for att in range(X.shape[1]):
                # self.bincuts[att, 1:-1] = np.histogram_bin_edges(X[:, att], bins=self.b, range=att_ranges[att])[1:-1]
                self.bincuts[att, :] = np.histogram_bin_edges(X[:, att], bins=self.b, range=att_ranges[att])
        elif self.bin_strategy == 'tasche':
            # only for binary quantification
            self.bincuts = np.zeros((X.shape[1], self.b+1))
            for att in range(X.shape[1]):
                mu = 0
                std = 0
                for n_cls, cls in enumerate(self.classes_):
                    mu = mu + np.mean(X[y == cls, att])
                    std = std + np.std(X[y == cls, att])
                mu = mu / n_classes
                std = std / n_classes
                if std > 0:
                    self.bincuts[att, :] = [std * norm.ppf(i/self.b) + mu for i in range(0, self.b+1)]
                else:
                    self.bincuts[att, :] = np.histogram_bin_edges(X[:, att], bins=self.b, range=[0, 0])


        self.train_dist_ = np.zeros((self.b * X.shape[1], n_classes))
        for n_cls, cls in enumerate(self.classes_):
            # compute pdf
            for att in range(X.shape[1]):
                self.train_dist_[att * self.b:(att + 1) * self.b, n_cls] = \
                    np.histogram(X[y == cls, att], bins=self.bincuts[att, :])[0]

            self.train_dist_[:, n_cls] = self.train_dist_[:, n_cls] / np.sum(y == cls)
        # except:
        #    print(self.bincuts)

    def predict(self, X, method='cc'):
        if not self.b:
            raise ValueError("If HDy predictions are in order, the quantifier must be trained with the parameter `b`")

        n_classes = len(self.classes_)

        # NEW
        test_dist = np.zeros((self.b * X.shape[1], 1))
        # compute pdf
        for att in range(X.shape[1]):
            test_dist[att * self.b:(att + 1) * self.b, 0] = np.histogram(X[:, att], bins=self.bincuts[att, :])[0]

        test_dist = test_dist / len(X)

        return solve_hd(self.train_dist_, test_dist, n_classes, solver="ECOS")
