import cvxpy
from numpy import linalg as la
import numpy as np
import math
import numbers
from scipy.optimize import linprog
from sklearn.utils import check_X_y
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

def nearest_pd(A):
    """Find the nearest positive-definite matrix to input

    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].

    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """

    B = (A + A.T) / 2
    _, s, V = la.svd(B)

    H = np.dot(V.T, np.dot(np.diag(s), V))

    A2 = (B + H) / 2

    A3 = (A2 + A2.T) / 2

    if is_pd(A3):
        return A3

    spacing = np.spacing(la.norm(A))
    # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
    # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
    # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
    # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
    # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
    # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
    # `spacing` will, for Gaussian random matrixes of small dimension, be on
    # othe order of 1e-16. In practice, both ways converge, as the unit test
    # below suggests.
    I = np.eye(A.shape[0])
    k = 1
    while not is_pd(A3):
        mineig = np.min(np.real(la.eigvals(A3)))
        A3 += I * (-mineig * k ** 2 + spacing)
        k += 1

    return A3


def is_pd(B):
    """Returns true when input is positive-definite, via Cholesky"""
    try:
        _ = la.cholesky(B)
        return True
    except la.LinAlgError:
        return False


invphi = (math.sqrt(5) - 1) / 2  # 1/phi
invphi2 = (3 - math.sqrt(5)) / 2  # 1/phi^2


def gss(f, tol=1e-5):
    """
    Golden section search.

    Given a function f with a single local minumum in
    the interval [a,b], gss returns a subset interval
    [c,d] that contains the minimum with d-c <= tol.
    """

    a, b = 0., 1.
    h = b - a
    if h <= tol: return (a, b)

    # required steps to achieve tolerance
    n = int(math.ceil(math.log(tol / h) / math.log(invphi)))

    c = a + invphi2 * h
    d = a + invphi * h
    yc = f(c)
    yd = f(d)

    for k in range(n - 1):
        if yc < yd:
            b = d
            d = c
            yd = yc
            h = invphi * h
            c = a + invphi2 * h
            yc = f(c)
        else:
            a = c
            c = d
            yc = yd
            h = invphi * h
            d = a + invphi * h
            yd = f(d)

    if yc < yd:
        return (a, d)
    else:
        return (c, b)


def solve_hd(train_dist, test_dist, n_classes, solver="ECOS"):
    p = cvxpy.Variable(n_classes)
    s = cvxpy.mul_elemwise(test_dist, (train_dist.T * p))
    objective = cvxpy.Minimize(1 - cvxpy.sum_entries(cvxpy.sqrt(s)))
    contraints = [cvxpy.sum_entries(p) == 1, p >= 0]

    prob = cvxpy.Problem(objective, contraints)
    prob.solve(solver=solver)
    return np.array(p.value).squeeze()

def solve_mmy(train_dist, test_dist, n_classes):
    c = np.hstack((np.ones(len(train_dist)),
                   np.zeros(n_classes)))

    A_ub = np.vstack((np.hstack((-np.eye(len(train_dist)), train_dist)),
                     np.hstack((-np.eye(len(train_dist)), -train_dist))
                      ))

    b_ub = np.vstack((test_dist, -test_dist))

    A_eq = np.hstack((np.zeros(len(train_dist)),
                   np.ones(n_classes)))
    A_eq = np.expand_dims(A_eq, axis=0)

    b_eq = 1

    x = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq)['x']

    p = x[-n_classes:]

    return p


def phdy_f(probs, weights, k):

    n = len(probs)
    quantils = np.zeros(k)

    idx_q = 0
    cumsum = 0

    for iw in range(n):
        cumsum += weights[iw]
        if cumsum >= n / k:
            diff = cumsum - n / k
            quantils[idx_q] += probs[iw] * (weights[iw] - diff)
            cumsum = diff
            idx_q += 1
            if idx_q == k - 1:
                quantils[idx_q] += probs[iw] * diff + np.sum(probs[iw+1:] * weights[iw+1:])
                break
            quantils[idx_q] += probs[iw] * diff
        else:
            quantils[idx_q] += probs[iw] * weights[iw]

    quantils = quantils / (n / k)

    return quantils


def create_bags_with_multiple_prevalence(X, y, n=1001, rng=None):


    if isinstance(rng, (numbers.Integral, np.integer)):
        rng=np.random.RandomState(rng)
    if not isinstance(rng, np.random.RandomState):
         raise ValueError("Invalid random generaror object: neg")

    X, y = check_X_y(X, y)
    classes = np.unique(y)
    n_classes = len(classes)
    m = len(X)

    for i in range(n):
        # Kraemer method:
        # http://www.cs.cmu.edu/~nasmith/papers/smith+tromble.tr04.pdf
        # http://blog.geomblog.org/2005/10/sampling-from-simplex.html
        low = 0   #min number of samples
        high = m   #max number of samples

        #to soft limits
        low = round(m * 0.05)
        high = round(m * 0.95)

        ps = rng.randint(low, high, n_classes - 1)
        ps = np.append(ps, [0, m]);
        ps = np.diff(np.sort(ps))  #number of samples for each class
        prev = ps / m  # to obtain prevalences
        #print("** prevalencias:",prev)
        idxs = []
        for n, p in zip(classes, ps.tolist()):
            if p!=0:
              idx = rng.choice(np.where(y == n)[0], p, replace=True)
              idxs.append(idx)

        idxs = np.concatenate(idxs)
        yield X[idxs], y[idxs], prev


def binary_kl_divergence(p_true, p_pred, eps=1e-12):
    """Also known as discrimination information, relative entropy or normalized cross-entropy
    (see [Esuli and Sebastiani 2010; Forman 2008]). KL Divergence is a special case of the family of f-divergences and
    it can be defined for binary quantification.

    Parameters
    ----------
    p_true : array_like, shape = (n_samples)
        True binary prevalences.

    p_pred : array_like, shape = (n_samples)
        Predicted binary prevalences.
    """

    kl = (p_true + eps) * np.log2((p_true + eps) / (p_pred + eps)) + (1 - p_true + eps) * np.log2(
        (1 - p_true + eps) / (1 - p_pred + eps))
    return kl


def absolute_error(p_true, p_pred):
    """Just the absolute difference between both prevalences.

        Parameters
        ----------
        p_true : array_like, shape=(n_classes)
            True prevalences. In case of binary quantification, this parameter could be a single float value.

        p_pred : array_like, shape=(n_classes)
            Predicted prevalences. In case of binary quantification, this parameter could be a single float value.
        """
    return np.abs(p_pred - p_true)


def tpr(clf, X, y):
    tn, fp, fn, tp = confusion_matrix(y, clf.predict(X)).ravel()
    tpr = tp / (tp + fn)
    return tpr


def fpr(clf, X, y):
    tn, fp, fn, tp = confusion_matrix(y, clf.predict(X)).ravel()
    fpr = fp / (fp + tn)
    return fpr


def g_mean(clf, X, y):
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y, clf.predict(X), labels=clf.classes_)
    fpr = cm[0, 1] / float(cm[0, 1] + cm[0, 0])
    tpr = cm[1, 1] / float(cm[1, 1] + cm[1, 0])
    return np.sqrt((1 - fpr) * tpr)



def normalize(X_train, X_test):
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test