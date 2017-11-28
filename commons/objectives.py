"""
Set of commonly used objective and evaluation functions I've been using in my work.

Author: Moshe Lichman
"""
from __future__ import division
import numpy as np


def _fast_log_factorial(x):
    """Fast approximate gammaln from paul mineiro.

    http://www.machinedlearnings.com/2011/06/faster-lda.html
    """
    x = x + 1
    logterm = np.log(x * (1.0 + x) * (2.0 + x))
    xp3 = 3.0 + x
    return -2.081061466 - x + 0.0833333 / xp3 - logterm + (2.5 + x) * np.log(xp3)


def pois_log_prob(target, est_lambda):
    """Fast computation of the Poisson probability log probability.

     Args
    ------
        1. target:      <(D, ) int>     target counts
        2. est_lambda:        <(D, ) float>   rate parameter

     Returns
    --------
        1. log_prob:    <(D, ) float>   log probability for each point
    """
    return target * np.log(est_lambda) - est_lambda - _fast_log_factorial(target)


"""

  The following metrics measure accuracy in terms of the predicted rate/counts value.
In my work I set this value as the expected rate according to each model.

"""


def _true_positive(target, pred):
    """True positive for counts data. """
    tmp = np.vstack([target, pred])
    return np.sum(np.min(tmp, axis=0))


def _prec(target, pred):
    """Precision for F1 Measure. """
    tp = _true_positive(target, pred)
    total_pos = np.sum(pred)

    return tp / total_pos


def _recall(target, pred):
    """Recall for F1 Measure. """
    tp = _true_positive(target, pred)
    total_pos = np.sum(target)

    return tp / total_pos


def f_measure(target, pred):
    """F1 Measure for counts data. """
    recall = _recall(target, pred)
    prec = _prec(target, pred)

    return 2 * (prec * recall) / (prec + recall)


def mse(target, pred):
    """Mean Squared Error. """
    tmp = pred - target
    return np.mean(np.power(tmp, 2))


def mae(target, pred):
    """Mean Average Error. """
    return np.mean(np.abs(target - pred))



