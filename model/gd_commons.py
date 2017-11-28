"""
Common methods that are shared across both regression models. One could put it as an Abstract class but I think this
is cleaner.

Author: Moshe Lichman
"""
from __future__ import division
import numpy as np

# This will always show complication error, just ignore it if you are certain that the file is there
from model import fast_methods as fm

from commons import time_measure as tm
from commons import async_sampler as asamp


def mul_feat_coeff(users, items, user_feat, coeff_global, coeff_users, coeff_items=None):
    """Multiplies the features table and the intercepts with the coefficients (smart dot product basically).

    This cannot be done with numpy efficiently because each user has a different set of coefficients that need
    to be matched, so all the code is in cython basically.

     Args
    ------
        1. users:       <(D, ) int>     user ids
        2. items:       <(D, ) int>     item ids
        3. user_feat:   <(D, f) float>  user features values
        4. coeff_global <float>         global shared intercept
        5. coeff_users  <(N, f) float>  user coefficients (including the intercept)
        6. coeff_items  <(M, ) float>   item intercepts. None for the exposure model in which there's no item intercept

     Returns
    ---------
        1. mul_sum: <(D, ) float> dot product for each row of the features and coefficients
    """
    # I found to be easier on the cython part if you create the data structure outside instead of using malloc and free
    # inside the cython code.
    mul_sum = np.zeros(user_feat.shape[0])

    if coeff_items is None:
        fm.mul_feat_coeff_no_items(users, user_feat, coeff_global, coeff_users, mul_sum)
    else:
        fm.mul_feat_coeff_with_items(users, items, user_feat, coeff_global, coeff_users, coeff_items, mul_sum)

    return mul_sum


def get_adam_update(step_size, grad, adam_values, b1=.95, b2=.999):
    """Computes an adaptive update step size using the AdaM algorithm.

     Args
    ------
        1. step_size:   <float>         initial step size. The algorithm is only sensitive to this value in levels of
                                        order of magnitude
        2. grad:        <? float>       current gradient (could be a value, a vector or a matrix)
        3. adam_values: <dict>          dictionary containint the relevant adam information
        4. b1:          <float>         weight of past mean
        5. b2:          <float>         weight of the past var

     Returns
    ---------
        1. The update step size
    """
    adam_values['t'] += 1

    # update mean
    adam_values['mean'] = b1 * adam_values['mean'] + (1 - b1) * grad
    m_hat = adam_values['mean'] / (1 - b1 ** adam_values['t'])

    # update variance
    adam_values['var'] = b2 * adam_values['var'] + (1 - b2) * grad ** 2
    v_hat = adam_values['var'] / (1 - b2 ** adam_values['t'])

    return step_size * m_hat / (np.sqrt(v_hat) + 1e-8)


def grad_for_user(users, d_pois_reg_user, d_user_prior):
    """Computes the gradient for \beta_i including the user intercept.

     Args
    ------
        1. users:               <(D, ) int>      user ids
        2. d_pois_reg_user:     <(D, f) float>   derivative of user features
        3. d_user_prior:        <(N, f) float>   derivative of the user coefficient prior


     Returns
    ---------
        1. grad:    <(N, f) float>   gradient for each user
    """
    # I found to be easier on the cython part if you create the data structure outside instead of using malloc and free
    # inside the cython code.
    user_counts = np.zeros(d_user_prior.shape[0])
    grad = np.zeros(d_user_prior.shape)

    point = tm.get_point('grad_for_user')

    fm.grad_for_user(users, d_pois_reg_user, d_user_prior, user_counts, grad)

    point.collect()

    return grad


def grad_for_item(items, d_pois_reg_item, d_item_prior):
    """Computes the gradient for the item intercept.

     Args
    ------
        1. items:               <(D, ) int>     item ids
        2. d_pois_reg_item:     <(D, f) float>  derivative of item intercept
        3. d_item_prior:        <(M, f) float>  derivative of the item coefficient prior


     Returns
    ---------
        1. grad:    <(M, ) float>   gradient for each item
    """
    # I found to be easier on the cython part if you create the data structure outside instead of using malloc and free
    # inside the cython code.
    item_counts = np.zeros(d_item_prior.shape[0])
    grad = np.zeros(d_item_prior.shape[0])

    point = tm.get_point('grad_for_item')

    fm.grad_for_item(items, d_pois_reg_item, d_item_prior, item_counts, grad)

    point.collect()

    return grad


def grad_for_global(d_pois_reg_global, d_global_prior):
    """Computes the gradient for the global intercept.

     Args
    ------
        1. d_pois_reg_global:       <(D, ) float>   derivative for global intercept
        2. d_global_prior:          <float>         derivative for the global specific coefficient prior

     Returns
    ---------
        1. grad:        <float>     gradient for the global specific coefficient
    """
    return np.mean(d_pois_reg_global) - d_global_prior


"""
  Fast sampling related code

To avoid the overhead of sampling a mini batch at each iteration (and there's a lof of it!) I do in
an asynch manner. For more details look at the code of the AsyncSampler class.
"""
# Starting a sampler
sampler = asamp.AsyncSampler(3)


def stop_sampler():
    """Stopping the sampler.

    This is needed for cases where this is not a one time run thing (like from IPython or a Notebook. Otherwise
    the sampler is still running in the background and can take a a lot of memory.
    """
    sampler.stop_sampler()


def fast_sample(num_points, batch_size):
    """Generates a choice sample of size batch_size from num_points.

     Args
    ------
        1. num_points:      <int>   number of points to choose from.
        2. btach_size:      <int>   number of points to sample.

     Returns
    ---------
        1. samp: <(batch_size, ) int>     indexes of selected points
    """
    point = tm.get_point('fast_sample_%d_%d' % (num_points, batch_size))

    samp = sampler.get_sample(num_points, batch_size)

    point.collect()

    return samp

