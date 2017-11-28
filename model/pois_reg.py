"""
Author: Moshe Lichman
"""
from __future__ import division
import numpy as np

from commons import log_utils as log
from commons import time_measure as tm
from commons import objectives

from model import gd_commons


class PoissonRegression(object):
    def __init__(self, N, M, beta_0=None, beta_u=None, beta_i=None, beta_gd_num_iter=100, beta_gd_tol=0.01,
                 beta_gd_lamb=0.01, beta_gd_step_size=0.01, beta_0_prior=0, beta_i_prior=0, beta_u_prior=0,
                 beta_gd_batch_size=10000, beta_gd_ll_iter=10,
                 min_gd_iter=100, gd_num_dec=np.inf, **kwargs):

        self.N, self.M = N, M
        self.beta_0, self.beta_u, self.beta_i = beta_0, beta_u, beta_i
        self.beta_0_prior, self.beta_i_prior, self.beta_u_prior = beta_0_prior, beta_i_prior, beta_u_prior
        self.trained = False

        # GD hyperparameters
        self.gd_lamb = beta_gd_lamb
        self.gd_num_iter = beta_gd_num_iter
        self.gd_tol = beta_gd_tol
        self.gd_batch_size = beta_gd_batch_size
        self.gd_step_size = beta_gd_step_size
        self.gd_ll_iters = beta_gd_ll_iter
        self.min_gd_iter = min_gd_iter
        self.gd_num_dec = gd_num_dec

    def get_est_lambda(self, users, items, user_feat):
        """Estimates the \lambda parameters.

        This code uses the current \beta values and estimates it for each user i and item j pairs in the users and items
        vectors according to the corresponding features.

         Args
        ------
            1. users:       <(D, ) int>      user ids
            2. items:       <(D, ) int>      item ids
            3. user_feat:   <(D, f) float>   user features values

         Returns
        ---------
            1. est_lamb:    <(D, ) ndarray of type float>   estimated lambdas.
        """
        point = tm.get_point('get_est_lambda')
        beta_x = gd_commons.mul_feat_coeff(users, items, user_feat, self.beta_0, self.beta_u, self.beta_i)
        est_lamb = np.exp(beta_x)
        point.collect()

        return est_lamb

    def _beta_derivative_vals(self, users, items, user_feat, target):
        """Computes the derivations for each element in the matrix.

        Note that this is not where the gradient is computed, but just where each element in the feature table is
        derived. This also includes the two intercept. If there's a prior, it is derived as well.

        The reason for the separation is because of the fixed-regression in which we have a fixed effect for population,
        each individual and each item so it is easier to first compute the derivation at each point using mat operations
        and later compute the different gradients separately.

         Args
        ------
            1. users:       <(D, ) int>     user ids
            2. items:       <(D, ) int>     item ids
            3. user_feat:    <(D, f) float> user features values.
            4. target:      <(D, ) int>     target rates

         Returns
        ---------
            1. d_pois_reg:       <(D, f + 2) float>  derivative of ALL features
            2. d_0_prior:        <float>             derivative of the global intercept prior
            3. d_i_prior:        <(M, ) float>       derivative of the item intercept prior
            3. d_u_prior:        <(N, f) float>      derivative of the user \beta (including intercept)
        """
        point = tm.get_point('pois_regression_deriv')

        # First computing \beta_u * features. It's going to be needed in the derivation computation.
        beta_u_x = gd_commons.mul_feat_coeff(users, items, user_feat, self.beta_0, self.beta_u, self.beta_i)

        # Adding two columns of ones for the 'non-user' feat. This is done to make the computation easier using matrix
        # operations.
        f_const = np.hstack([np.ones([user_feat.shape[0], 2]), user_feat])

        # Computing the parts of the Poisson regression derivative
        d_features = f_const * np.atleast_2d(np.exp(beta_u_x)).T
        d_target = f_const * np.atleast_2d(target).T
        d_pois_reg = d_target - d_features

        # The dervation of the prior
        d_0_prior = self.gd_lamb * (self.beta_0 - self.beta_0_prior)
        d_i_prior = self.gd_lamb * (self.beta_i - self.beta_i_prior)
        d_u_prior = self.gd_lamb * (self.beta_u - self.beta_u_prior)

        point.collect()

        return d_pois_reg, d_0_prior, d_i_prior, d_u_prior

    def _pois_reg_data_log_like(self, target, users, items, user_feat, weights=None):
        """Computes the data log likelihood.

         Args
        ------
            1. users:       <(D, ) int>     user ids
            2. items:       <(D, ) int>     item ids
            3. user_feat:   <(D, f) float>  data-driven (non-intercept) and user const features
            4. target:      <(D, ) int>     target rates
            5. weights:     <(D, ) float>   points weights for the weighted regression case

         Returns
        ---------
            1. <float> average data log likelihood.
        """
        point = tm.get_point('pois_reg_data_log_like')

        est_lambda = self.get_est_lambda(users, items, user_feat)
        curr_ll = objectives.pois_log_prob(target, est_lambda)

        if weights is not None:
            # Adjusting the weights.
            curr_ll *= weights

        point.collect()

        return np.mean(curr_ll)

    def learn_model(self, users, items, data_feat, target, weights=None):
        """Performs gradient decent optimziation to learn values of all \betas.

         Args
        ------
            1. users:       <(D, ) int>     user ids
            2. items:       <(D, ) int>     item ids
            3. data_feat:   <(D, f) float>  data-driven (non-intercept) features
            4. target:      <(D, ) int>     target rates
            5. weights:     <(D, ) float>   points weights for the weighted regression case

         Raise
        ------
            1. ValueError if any of the inputs are not of shape[0] == D
            2. ValueError if data_feat is not a 2d matrix.
            3. ValueError if weights is not None and not size D.
            4. Bunch of type and value errors if you did not follow the args input instruction.
        """
        if not users.shape[0] == items.shape[0] == data_feat.shape[0] == target.shape[0]:
            raise ValueError('All inputs must be same size D.')
        if data_feat.ndim != 2:
            raise ValueError('Data driven features must be 2d')
        if weights is not None and weights.shape[0] != users.shape[0]:
            raise ValueError('Weights inputs must be same size as the other inputs.')

        # Saving the user ids that this model used in training. This is used later in the test and validation to filter
        # out the users we've never seen before. This is crucial for online training when users can "join" the data at
        # later time windows. If in your pre-processing of the data you made sure that all users are in all time windows
        # this will not be of any use to you.
        self.trained_users = np.unique(users)

        # Adding the user intercept constant. In my code, the exposure process has different constants than the
        # rate process, so I only pass the data_feat to the methods and deal with the constants separately.
        # The reason I keep the user const in the user_feat is to avoid starting the counts from 1 in the cython code.
        # Other you trust me, or you can go and look at it :)
        user_feat = np.hstack([np.ones([data_feat.shape[0], 1]), data_feat])

        self._learn_beta(users, items, user_feat, target, weights)

        # Indicating the the model was trained. It's not really necessary, just good practice if you ask me :)
        self.trained = True

    def _initialize_beta(self, f):
        """Makes sure all \beta's are initialized. If not sample from 0 mean normal distribution.

         Args
        ------
            1. f: <int> number of data-driven (non-intercepts) features + the user intercept.
        """
        if self.beta_u is None:
            self.beta_u = np.random.normal(0, 0.1, [self.N, f])
        if self.beta_i is None:
            self.beta_i = np.random.normal(0, 0.1, self.M)
        if self.beta_0 is None:
            self.beta_0 = np.random.normal(0, 0.1, 1)[0]

    def predict(self, users, items, data_feat):
        """Predicts the expected rate value.

         Args
        ------
            1. users:       <(D, ) int>     user ids
            2. items:       <(D, ) int>     item ids
            3. data_feat:   <(D, f) float>  data-driven (non-intercept) features

         Returns
        ---------
            1. est_lambda: <(D, ) float>    expected rate value
        """
        user_feat = np.hstack([np.ones([data_feat.shape[0], 1]), data_feat])
        est_lambda = self.get_est_lambda(users, items, user_feat)
        return est_lambda

    def _learn_beta(self, users, items, user_feat, target, weights=None):
        """Learns all the \beta's using stochastic gradient descent with ADAM.

         Args
        ------
            1. users:       <(D, ) int>     user ids
            2. items:       <(D, ) int>     item ids
            3. user_feat:   <(D, f) float>  user features values
            4. target:      <(D, ) int>     target rates
            5. weights:     <(D, ) float>   points weights for the weighted regression case

         Raises
        --------
            1. ValueError if coefficients went out of hand and got the value of np.inf.
        """
        self._initialize_beta(user_feat.shape[1])

        # ADAM initial values
        adam_vals_u = {'mean': np.zeros(self.beta_u.shape), 'var': np.zeros(self.beta_u.shape), 't': 0}
        adam_vals_i = {'mean': np.zeros(self.beta_i.shape), 'var': np.zeros(self.beta_i.shape), 't': 0}
        adam_vals_0 = {'mean': 0, 'var': 0, 't': 0}

        # Number of times the likelihood went down. Used to prevent overfitting and parameter explosion.
        num_down = 0

        prev_ll = curr_ll = -np.inf
        reached_conv = False

        # Gradient descent main loop
        for i in range(1, self.gd_num_iter + 1):
            # Sampling a mini-bucket
            samp = gd_commons.fast_sample(user_feat.shape[0], self.gd_batch_size)

            point = tm.get_point('pois_reg_sgd_iter')  # Taking this time point after the sample.

            # First computing all the derivative values. Not computing the gradients yet.
            d_pois_reg, d_0_prior, d_i_prior, d_u_prior = \
                self._beta_derivative_vals(users[samp], items[samp], user_feat[samp], target[samp])

            if weights is not None:
                # It's weighted regression and I need to modify the weight of each point.
                d_pois_reg *= np.atleast_2d(weights[samp]).T

            # Computing all the gradients
            g_grad = gd_commons.grad_for_global(d_pois_reg[:, 0], d_0_prior)
            i_grad = gd_commons.grad_for_item(items[samp], d_pois_reg[:, 1], d_i_prior)
            u_grad = gd_commons.grad_for_user(users[samp], d_pois_reg[:, 2:], d_u_prior)

            # These operations are safe because if the user or item were not in the sample the grad for them will be
            # zero.

            self.beta_0 += gd_commons.get_adam_update(self.gd_step_size, g_grad, adam_vals_0)
            self.beta_i += gd_commons.get_adam_update(self.gd_step_size, i_grad, adam_vals_i)
            self.beta_u += gd_commons.get_adam_update(self.gd_step_size, u_grad, adam_vals_u)

            point.collect()

            # Checking for convergence - using only the data likelihood.
            if i > self.min_gd_iter and i % self.gd_ll_iters == 0:
                curr_ll = self._pois_reg_data_log_like(target, users, items, user_feat, weights)

                if curr_ll < prev_ll:
                    num_down += 1

                if np.isnan(curr_ll) or np.isinf(curr_ll):
                    raise ValueError('Pois_Reg: Coefficient values went out of hand -- adjust regularizer value.')

                log.info('Pois_Reg data log like: [%.3f --> %.3f]' % (prev_ll, curr_ll))

                if np.abs(curr_ll - prev_ll) <= self.gd_tol or num_down >= self.gd_num_dec:
                    log.info('Pois_Reg: Reached convergance after %d iterations' % i)

                    reached_conv = True
                    break

                prev_ll = curr_ll

        if not reached_conv:
            log.error('Pois_Reg: Did not reach convergence after %d iterations' % self.gd_num_iter)

        log.info('Pois_Reg: Train log like %.3f' % curr_ll)

