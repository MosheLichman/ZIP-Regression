"""
Author: Moshe Lichman
"""
from __future__ import division
import numpy as np
from os.path import join

from commons import helpers
from commons import log_utils as log
from commons import time_measure as tm
from commons import file_utils as flu
from commons import objectives

from models import gd_commons


class PoissonRegression(object):
    def __init__(self, N, M, beta_0=None, beta_u=None, beta_i=None, beta_gd_num_iter=100, beta_gd_tol=0.01,
                 beta_gd_lamb=0.01, beta_gd_step_size=0.01, beta_gd_emp_prior=False,
                 beta_gd_decay=False, beta_gd_batch_size=10000, num_proc=1, beta_gd_ll_iter=10,
                 beta_gd_adam=False, beta_gd_weight_sample=False, **kwargs):
        """Instantiate a new Poisson Regression model.

         Args
        ------

        N: <int> Number of rows (users)
        M: <int> Number of columns (items)
        beta_0: <float>  Globally shared Beta coefficient


        :param M:
        :param beta_0:
        :param beta_u:
        :param beta_i:
        :param beta_gd_num_iter:
        :param beta_gd_tol:
        :param beta_gd_lamb:
        :param beta_gd_step_size:
        :param beta_gd_emp_prior:
        :param beta_gd_decay:
        :param beta_gd_batch_size:
        :param num_proc:
        :param beta_gd_ll_iter:
        :param beta_gd_adam:
        :param beta_gd_weight_sample:
        :param kwargs:
        :return:
        """

        self.N, self.M = N, M
        self.beta_0, self.beta_u, self.beta_i = beta_0, beta_u, beta_i
        self.prev_ll = -np.inf
        self.trained = False

        # GD hyperparameters
        self.gd_lamb = beta_gd_lamb
        self.gd_num_iter = beta_gd_num_iter
        self.gd_tol = beta_gd_tol
        self.gd_decay = beta_gd_decay
        self.gd_batch_size = beta_gd_batch_size
        self.gd_step_size = beta_gd_step_size
        self.gd_emp_prior = beta_gd_emp_prior
        self.decay = 1
        self.num_proc = num_proc
        self.gd_ll_iters = beta_gd_ll_iter
        self.gd_adam = beta_gd_adam
        self.gd_weights_sample = beta_gd_weight_sample

    def get_est_lambda(self, users, items, features):
        """Estimates the \lambda parameters.

        This code uses the current \beta values and estimates it for each user i and item j pairs in the users and items
        vectors according to the corresponding features.

         Args
        ------
            1. users:       <(N, ) ndarray of type int>     user ids
            2. items:       <(N, ) ndarray of type int>     item ids
            3. features:    <(N, d) ndarray of type float>  features values.

         Returns
        ---------
            1. est_lamb:    <(N, ) ndarray of type float>   estimated lambdas.
        """
        if users.shape[0] != items.shape[0] or users.shape[0] != features.shape[0]:
            raise AssertionError('Numbers of users, items and features have to be the same.')

        point = tm.get_point('get_est_lambda')
        beta_x = gd_commons.mul_feat_coeff(users, items, features, self.beta_0, self.beta_u, self.beta_i,
                                           num_proc=self.num_proc)
        est_lamb = np.exp(beta_x)
        point.collect()

        return est_lamb

    def _beta_derivative_vals(self, users, items, features, target):
        """Computes the derivations for each element in the matrix.

        Note that this is not where the gradient is computed, but just where each element in the feature table is
        derived. This also includes the two intercept. If there's a prior, it is derived as well.

        The reason for the separation is because of the fixed-regression in which we have a fixed effect for population,
        each individual and each item so it is easier to first compute the derivation at each point using mat operations
        and later compute the different gradients separately.

         Args
        ------
            1. users:       <(N, ) ndarray of type int>     user ids
            2. items:       <(N, ) ndarray of type int>     item ids
            3. features:    <(N, d) ndarray of type float>  features values.
            4. target:      <(N, ) ndarray of type int>     target rates

         Returns
        ---------
            1. d_features:       <(N, d + 2) ndarray of type float>  derivative of features
            2. d_0_prior:        <float>                             derivative of the glob-shared intercept prior
            3. d_i_prior:        <
        """
        # Computing \theta * X
        num_proc = self.num_proc if self.gd_weights_sample else 1
        # If it's the weight sampling then it's going to be a lot of points.
        beta_x = gd_commons.mul_feat_coeff(users, items, features, self.beta_0, self.beta_u, self.beta_i,
                                           num_proc=num_proc)

        # The rest of the gradient using theta_x

        # Adding two columns of ones for the 'non-user' feat. This is done to make the computation easier using matrix
        # operations.
        tmp = np.hstack([np.ones([features.shape[0], 2]), features])
        right = tmp * np.atleast_2d(np.exp(beta_x)).T
        left = tmp * np.atleast_2d(target).T
        d_features = left - right

        # The derivative of the prior
        if self.gd_emp_prior:
            # Gaussian prior centered around the global mean
            u_prior = np.mean(self.beta_u, axis=0)
            i_prior = np.mean(self.beta_i)
        else:
            # Gaussian prior centered around 0
            u_prior = 0
            i_prior = 0

        d_0_prior = self.gd_lamb * self.beta_0  # 0 prior
        d_i_prior = self.gd_lamb * (self.beta_i - i_prior)
        d_u_prior = self.gd_lamb * (self.beta_u - u_prior)

        return d_features, d_0_prior, d_i_prior, d_u_prior

    def _mle(self, target, users, items, features, weights=None):
        point = tm.get_point('beta_mle_est_lambda')
        est_lambda = self.get_est_lambda(users, items, features)
        point.collect()

        point = tm.get_point('beta_mle_log_factorial')
        y_log_fact = helpers.log_factorial(target)
        point.collect()

        point = tm.get_point('beta_mle_curr_ll_numpy')
        curr_ll = (target * np.log(est_lambda)) - est_lambda - y_log_fact
        if weights is not None:
            curr_ll *= weights

        point.collect()

        return np.mean(curr_ll)

    def learn_model(self, users, items, data_feat, target, weights=None):
        self.trained_users = np.unique(users)
        features = np.hstack([np.ones([data_feat.shape[0], 1]), data_feat])
        try:
            self._learn_beta(users, items, features, target, weights)
            self.trained = True
        except Exception as e:
            log.error(e.message)

    def _learn_beta(self, users, items, features, target, weights=None):
        # If any of the parameters wasn't initialized
        if self.beta_u is None:
            self.beta_u = np.random.normal(0, 0.1, [self.N, features.shape[1]])
        if self.beta_i is None:
            self.beta_i = np.random.normal(0, 0.1, self.M)
        if self.beta_0 is None:
            self.beta_0 = np.random.normal(0, 0.1, 1)[0]

        if self.gd_adam:
            adam_vals_u = {'mean': np.zeros(self.beta_u.shape), 'var': np.zeros(self.beta_u.shape), 't': 0}
            adam_vals_i = {'mean': np.zeros(self.beta_i.shape), 'var': np.zeros(self.beta_i.shape), 't': 0}
            adam_vals_0 = {'mean': 0, 'var': 0, 't': 0}

        # Computing the lambda array

        reached_conv = False
        for i in range(1, self.gd_num_iter + 1):
            beta_iter_point = tm.get_point('beta_sgd_iter')
            point = tm.get_point('beta_sgd_samp')
            if self.gd_weights_sample:
                samp = gd_commons.fast_sample_with_weights(weights)
            else:
                samp = gd_commons.fast_sample(features.shape[0], self.gd_batch_size)

            point.collect()

            point = tm.get_point('beta_derivative_vals')
            d_mle, d_g_prior, d_i_prior, d_u_prior = \
                self._beta_derivative_vals(users[samp], items[samp], features[samp], target[samp])

            point.collect()

            # TODO: Discuss the most proper way to combine the weights and the prior/regularization with Padhraic
            if weights is not None and not self.gd_weights_sample:
                # If it's weight sample no need to modify the mle with the weights
                d_mle *= np.atleast_2d(weights[samp]).T

            # Updating the gradient
            g_grad = gd_commons.grad_for_global(d_mle[:, 0], d_g_prior)
            i_grad = gd_commons.grad_for_item(items[samp], d_mle[:, 1], d_i_prior)
            u_grad = gd_commons.grad_for_user(users[samp], d_mle[:, 2:], d_u_prior)

            a = self.gd_step_size / self.decay if self.gd_decay else self.gd_step_size

            # These operations are safe because if the user or item were not in the sample the grad for them will be
            # zero.
            point = tm.get_point('beta_grad_updates')
            if self.gd_adam:
                self.beta_0 += gd_commons.get_AdaM_update(a, g_grad, adam_vals_0)
                self.beta_i += gd_commons.get_AdaM_update(a, i_grad, adam_vals_i)
                self.beta_u += gd_commons.get_AdaM_update(a, u_grad, adam_vals_u)
            else:
                self.beta_0 += g_grad * a
                self.beta_i += i_grad * a
                self.beta_u += u_grad * a

            point.collect()

            beta_iter_point.collect()
            if i % self.gd_ll_iters == 0:
                point = tm.get_point('beta_mle')
                curr_ll = self._mle(target, users, items, features, weights)
                point.collect()
                if np.isnan(curr_ll) or np.isinf(curr_ll):
                    raise ValueError('Coefficient values went out of hand -- adjust lambda and/or step size')
                log.info('BETA GD MLE: [%.3f --> %.3f]' % (self.prev_ll, curr_ll))
                if np.abs(curr_ll - self.prev_ll) <= self.gd_tol:
                    log.info('BETA GD: Reached convergance after %d iterations' % i)
                    reached_conv = True
                    self.prev_ll = curr_ll
                    break
                else:
                    self.prev_ll = curr_ll

                self.decay += 1

        if not reached_conv:
            log.error('BETA GD: Did not reach convergance after %d iterations' % self.gd_num_iter)

        log.info('BETA GD: Train log like %.3f' % curr_ll)

    def test_log_prob(self, users, items, data_feat, target):
        if not self.trained:
            return -np.inf

        # At optimization time - it is very likely that some users don't have train data (because they're not active
        # yet). This makes sure that I'm not testing on them.
        test_users = np.unique(users)
        trained_user_mask = np.where(np.in1d(users, self.trained_users, assume_unique=False))
        log.info('Trained on %d out of %d test users' % (self.trained_users.shape[0], test_users.shape[0]))

        lambda_est = self.predict(users[trained_user_mask], items[trained_user_mask], data_feat[trained_user_mask])
        return objectives.pois_log_prob(target[trained_user_mask], lambda_est).mean()

    def test_abs_error(self, users, items, data_feat, target):
        if not self.trained:
            return np.inf

        # At optimization time - it is very likely that some users don't have train data (because they're not active
        # yet). This makes sure that I'm not testing on them.
        test_users = np.unique(users)
        trained_user_mask = np.where(np.in1d(users, self.trained_users, assume_unique=False))
        log.info('Trained on %d out of %d test users' % (self.trained_users.shape[0], test_users.shape[0]))

        lambda_est = self.predict(users[trained_user_mask], items[trained_user_mask], data_feat[trained_user_mask])
        return np.mean(np.abs(lambda_est - target[trained_user_mask]))

    def predict(self, users, items, data_feat):
        features = np.hstack([np.ones([data_feat.shape[0], 1]), data_feat])
        est_lambda = self.get_est_lambda(users, items, features)
        return est_lambda

    def save_model(self, path):
        # Saving the zip_model eta - we can estimate the pies from it

        # Saving the pos_mode beta - we can estimate the lambda from it
        flu.np_save(path, 'pos_beta_0.npy', self.beta_0)
        flu.np_save(path, 'pos_beta_i.npy', self.beta_i)
        flu.np_save(path, 'pos_beta_u.npy', self.beta_u)

    @staticmethod
    def load_model(path, num_proc):
        log.info('Loading ZIP model from path %s with %d num proc' % (path, num_proc))

        beta_0 = flu.np_load(join(path, 'pos_beta_0.npy'))
        beta_0 = np.atleast_1d(beta_0)[0]

        beta_i = flu.np_load(join(path, 'pos_beta_i.npy'))
        beta_u = flu.np_load(join(path, 'pos_beta_u.npy'))

        # TODO(MOSHE): Why do I need num_proc here??? And how do I deal with no N and M?
        # TODO(MOSHE): Specifically M because N can be taken from the coefficients
        return PoissonRegression(beta_0=beta_0, beta_i=beta_i, beta_u=beta_u, num_proc=num_proc)


def test_random_data():
    """All features are random and all targets are the same.

    In this test I want to see that the global +item + user coefficient are taking care of the counts
    """
    num_users, num_items = 10, 3
    data_feat = np.random.rand(2000, 3)
    mask = np.where(data_feat > 0.5)
    data_feat[mask[0], mask[1]] -= 1

    users = np.zeros(2000)
    for i in range(10):
        users[i * 200:(i + 1) * 200] = i

    items = np.arange(2000) % 3
    targets = np.ones(2000) * 100

    model = PoissonRegression(num_users, num_items, beta_gd_lamb=5, beta_gd_tol=0.001, beta_gd_decay=False,
                              beta_gd_num_iter=20000)

    features = np.hstack([np.ones([data_feat.shape[0], 1]), data_feat])
    model.learn_model(users.astype(np.intc), items.astype(np.intc), features, targets, )

    return model


def test_item_feature():
    """All features are random and all targets are the same for a given item.

    In this test I want to see that the item coefficient is taking care of the counts
    """
    num_users, num_items = 10, 3
    data_feat = np.random.rand(2000, 3)
    mask = np.where(data_feat > 0.5)
    data_feat[mask[0], mask[1]] -= 1

    users = np.zeros(2000)
    for i in range(10):
        users[i * 200:(i + 1) * 200] = i

    items = np.arange(2000) % 3
    targets = (np.arange(2000) % 3) * 100

    model = PoissonRegression(num_users, num_items, beta_gd_lamb=5, beta_gd_tol=0.001, beta_gd_decay=False,
                              beta_gd_num_iter=20000)

    features = np.hstack([np.ones([data_feat.shape[0], 1]), data_feat])
    model.learn_model(users.astype(np.intc), items.astype(np.intc), features, targets)

    return model


if __name__ == '__main__':
    # m = test_item_feature()
    m = test_random_data()
